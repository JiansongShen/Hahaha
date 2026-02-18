// Copyright (c) 2025-2026 Contributors of Hahaha(https://github.com/Napbad/Hahaha)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Contributors:
// jiansongshen (jason.shen111@outlook.com)
//
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "common/definitions.h"
#include "ml/dataset/DatasetHandleBlankStrategy.h"
#include "ml/dataset/DatasetInner.h"
#include "ml/dataset/DatasetInnerLoader.h"

namespace fs = std::filesystem;

using hahaha::ml::DatasetHandleBlankStrategy;
using hahaha::ml::DatasetInner;
using hahaha::ml::DatasetInnerLoader;
using hahaha::common::f32;
using hahaha::common::f64;
using hahaha::common::i32;

// Write content to a uniquely-named temp file, return the path.
static std::string writeTempCSV(const std::string& content,
                                const std::string& tag = "default") {
    auto path = (fs::temp_directory_path() /
                 ("hahaha_test_" + tag + ".csv")).string();
    std::ofstream ofs(path, std::ios::trunc);
    ofs << content;
    return path;
}

// Load a 3-row × 2-col float dataset: [[1,2],[3,4],[5,6]]
template <typename T>
static DatasetInner<T> make3x2Dataset(const std::string& tag = "") {
    const std::string csv = "col1,col2\n1.0,2.0\n3.0,4.0\n5.0,6.0\n";
    DatasetInner<T> ds;
    DatasetInnerLoader().loadFromCSVTo<T>(
        writeTempCSV(csv, "3x2_" + tag + typeid(T).name()), ds);
    return ds;
}

// --- DatasetInner: empty dataset --------------------------------------------

class DatasetEmptyTest : public ::testing::Test {
  protected:
    DatasetInner<f32> ds_; // default-constructed, no samples
};

TEST_F(DatasetEmptyTest, SizeIsZero) {
    // samples_.getDimensions() == 0 branch → returns 0
    EXPECT_EQ(ds_.size(), 0u);
}

TEST_F(DatasetEmptyTest, BeginEqualsEnd) {
    // end() empty-dataset guard → iterator(this, 0) == begin()
    EXPECT_EQ(ds_.begin(), ds_.end());
}

TEST_F(DatasetEmptyTest, ConstBeginEqualsConstEnd) {
    const DatasetInner<f32>& cds = ds_;
    EXPECT_EQ(cds.begin(), cds.end());
}

TEST_F(DatasetEmptyTest, CBeginEqualsCEnd) {
    EXPECT_EQ(ds_.cbegin(), ds_.cend());
}

TEST_F(DatasetEmptyTest, RangeForBodyNeverExecutes) {
    int count = 0;
    for ([[maybe_unused]] auto& row : ds_)
        ++count;
    EXPECT_EQ(count, 0);
}

TEST_F(DatasetEmptyTest, ParameterConstructorFillsMetadata) {
    DatasetInner<f32> ds({"f1", "f2"}, "my_dataset", {"label"});
    EXPECT_EQ(ds.getDatasetName(), "my_dataset");
    ASSERT_EQ(ds.getFeatures().size(), 2u);
    EXPECT_EQ(ds.getFeatures()[0], "f1");
    ASSERT_EQ(ds.getLabels().size(), 1u);
    EXPECT_EQ(ds.getLabels()[0], "label");
}

// --- DatasetInnerLoader: path / CSV error paths ----------------------------

class DatasetLoaderErrorTest : public ::testing::Test {};

TEST_F(DatasetLoaderErrorTest, MissingFileThrows) {
    EXPECT_THROW(DatasetInnerLoader::checkPathExist("/no/such/file.csv"),
                 std::invalid_argument);
}

TEST_F(DatasetLoaderErrorTest, ExistingFileDoesNotThrow) {
    auto path = writeTempCSV("col\n1.0\n", "exist");
    EXPECT_NO_THROW(DatasetInnerLoader::checkPathExist(path));
}

TEST_F(DatasetLoaderErrorTest, LoadNonExistentFileThrows) {
    DatasetInner<f32> ds;
    DatasetInnerLoader loader;
    EXPECT_THROW(loader.loadFromCSVTo<f32>("/nonexistent/path.csv", ds),
                 std::invalid_argument);
}

TEST_F(DatasetLoaderErrorTest, ColumnCountMismatchSkipsLine) {
    // line with wrong column count is skipped; rest of rows load normally
    const std::string csv = "col1,col2\n1.0,2.0\n3.0\n5.0,6.0\n";
    DatasetInner<f32> ds;
    DatasetInnerLoader().loadFromCSVTo<f32>(writeTempCSV(csv, "mismatch"), ds);
    EXPECT_EQ(ds.size(), 2u);
}

TEST_F(DatasetLoaderErrorTest, EmptyDataSectionProducesEmptyDataset) {
    // dataList.empty() → fillData assigns empty Tensor
    const std::string csv = "col1,col2\n";
    DatasetInner<f32> ds;
    DatasetInnerLoader().loadFromCSVTo<f32>(writeTempCSV(csv, "empty_data"), ds);
    EXPECT_EQ(ds.size(), 0u);
}

TEST_F(DatasetLoaderErrorTest, UnparseableValueSkipsLine) {
    // StringUtils::to<T> returns nullopt → line skipped
    const std::string csv = "col1,col2\n1.0,2.0\nNOT_A_NUMBER,4.0\n5.0,6.0\n";
    DatasetInner<f32> ds;
    DatasetInnerLoader().loadFromCSVTo<f32>(writeTempCSV(csv, "bad_val"), ds);
    EXPECT_EQ(ds.size(), 2u);
}

// --- DatasetInnerLoader: happy path ----------------------------------------

class DatasetLoaderTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ds_ = make3x2Dataset<f32>("loader_");
    }
    DatasetInner<f32> ds_;
};

TEST_F(DatasetLoaderTest, SizeMatchesRowCount) {
    // samples_.getDimensions() != 0 → returns shape[0]
    EXPECT_EQ(ds_.size(), 3u);
}

TEST_F(DatasetLoaderTest, ColumnNamesCorrect) {
    ASSERT_EQ(ds_.getFeatures().size(), 2u);
    EXPECT_EQ(ds_.getFeatures()[0], "col1");
    EXPECT_EQ(ds_.getFeatures()[1], "col2");
}

TEST_F(DatasetLoaderTest, DatasetNameIsFilePath) {
    EXPECT_FALSE(ds_.getDatasetName().empty());
}

TEST_F(DatasetLoaderTest, ColumnNamesWithSurroundingSpacesAreTrimmed) {
    const std::string csv = " col1 , col2 \n1.0,2.0\n";
    DatasetInner<f32> ds;
    DatasetInnerLoader().loadFromCSVTo<f32>(writeTempCSV(csv, "col_trim"), ds);
    ASSERT_EQ(ds.getFeatures().size(), 2u);
    EXPECT_EQ(ds.getFeatures()[0], "col1");
    EXPECT_EQ(ds.getFeatures()[1], "col2");
}

TEST_F(DatasetLoaderTest, GetItemReturnsCorrectRow) {
    auto row0 = ds_.getItem(0);
    EXPECT_EQ(row0.getDimensions(), 1u);
    EXPECT_EQ(row0.getShape()[0], 2u);
    EXPECT_NEAR(row0.at({0}), 1.0f, 1e-5f);
    EXPECT_NEAR(row0.at({1}), 2.0f, 1e-5f);

    auto row2 = ds_.getItem(2);
    EXPECT_NEAR(row2.at({0}), 5.0f, 1e-5f);
    EXPECT_NEAR(row2.at({1}), 6.0f, 1e-5f);
}

// --- DatasetInnerLoader: blank-cell strategies ------------------------------

class DatasetBlankStrategyTest : public ::testing::Test {
  protected:
    DatasetInnerLoader loader_;

    DatasetInner<f32> loadFloat(const std::string& csv,
                                const std::string& tag,
                                DatasetHandleBlankStrategy strategy) {
        loader_.setHandleBlankStrategy(strategy);
        DatasetInner<f32> ds;
        loader_.loadFromCSVTo<f32>(writeTempCSV(csv, tag), ds);
        return ds;
    }
};

TEST_F(DatasetBlankStrategyTest, UseZeroFillsBlankCell) {
    // isBlank(str) && UseZero → T(0)
    const std::string csv = "col1,col2\n1.0,\n3.0,4.0\n";
    auto ds = loadFloat(csv, "use_zero", DatasetHandleBlankStrategy::UseZero);
    ASSERT_EQ(ds.size(), 2u);
    auto row0 = ds.getItem(0);
    EXPECT_NEAR(row0.at({0}), 1.0f, 1e-5f);
    EXPECT_NEAR(row0.at({1}), 0.0f, 1e-5f);
}

TEST_F(DatasetBlankStrategyTest, SetNanFillsBlankCellWithNaN_Float) {
    // isBlank(str) && SetNan && isLegalFloatType<T> → quiet_NaN
    const std::string csv = "col1,col2\n1.0,\n3.0,4.0\n";
    auto ds = loadFloat(csv, "set_nan", DatasetHandleBlankStrategy::SetNan);
    ASSERT_EQ(ds.size(), 2u);
    EXPECT_TRUE(std::isnan(static_cast<float>(ds.getItem(0).at({1}))));
}

TEST_F(DatasetBlankStrategyTest, SetNanOnIntegerTypeSkipsLine) {
    // isBlank(str) && SetNan && !isLegalFloatType<T> (constexpr else) → unexpected → line skipped
    const std::string csv = "col1,col2\n1,\n3,4\n";
    loader_.setHandleBlankStrategy(DatasetHandleBlankStrategy::SetNan);
    DatasetInner<i32> ds;
    loader_.loadFromCSVTo<i32>(writeTempCSV(csv, "set_nan_int"), ds);
    EXPECT_EQ(ds.size(), 1u);
}

TEST_F(DatasetBlankStrategyTest, JumpOneSkipsLineWithBlank) {
    // isBlank(str) && JumpOne → unexpected (no error log) → line skipped
    const std::string csv = "col1,col2\n1.0,\n3.0,4.0\n5.0,6.0\n";
    auto ds = loadFloat(csv, "jump_one", DatasetHandleBlankStrategy::JumpOne);
    EXPECT_EQ(ds.size(), 2u);
    EXPECT_NEAR(ds.getItem(0).at({0}), 3.0f, 1e-5f);
    EXPECT_NEAR(ds.getItem(0).at({1}), 4.0f, 1e-5f);
}

// --- DatasetInner: mutable iterator ----------------------------------------

class DatasetIteratorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ds_ = make3x2Dataset<f32>("iter_");
    }
    DatasetInner<f32> ds_;
};

TEST_F(DatasetIteratorTest, BeginNotEqualEnd) {
    EXPECT_NE(ds_.begin(), ds_.end());
}

TEST_F(DatasetIteratorTest, DereferenceReturnsFirstRow) {
    auto it = ds_.begin();
    auto& row = *it;
    EXPECT_EQ(row.getDimensions(), 1u);
    EXPECT_NEAR(row.at({0}), 1.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, ArrowReturnsPointerToCurrentRow) {
    EXPECT_EQ(ds_.begin()->getDimensions(), 1u);
}

TEST_F(DatasetIteratorTest, PreIncrement) {
    auto it = ds_.begin();
    auto& ret = ++it;
    EXPECT_EQ(&ret, &it);
    EXPECT_NEAR((*it).at({0}), 3.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, PostIncrement) {
    auto it = ds_.begin();
    auto old = it++;
    EXPECT_NEAR((*old).at({0}), 1.0f, 1e-5f);
    EXPECT_NEAR((*it).at({0}), 3.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, PreDecrement) {
    auto it = ds_.end();
    auto& ret = --it;
    EXPECT_EQ(&ret, &it);
    EXPECT_NEAR((*it).at({0}), 5.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, PostDecrement) {
    auto it = ds_.end();
    auto old = it--;
    EXPECT_EQ(old, ds_.end());
    EXPECT_NEAR((*it).at({0}), 5.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, PlusEqualsN) {
    auto it = ds_.begin();
    it += 2;
    EXPECT_NEAR((*it).at({0}), 5.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, OperatorPlusN) {
    EXPECT_NEAR((*(ds_.begin() + 1)).at({0}), 3.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, MinusEqualsN) {
    auto it = ds_.end();
    it -= 1;
    EXPECT_NEAR((*it).at({0}), 5.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, OperatorMinusN) {
    EXPECT_NEAR((*(ds_.end() - 2)).at({0}), 3.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, IteratorDifference) {
    EXPECT_EQ(ds_.end() - ds_.begin(), 3);
}

TEST_F(DatasetIteratorTest, SubscriptOperator) {
    auto row1 = ds_.begin()[1];
    EXPECT_NEAR(row1.at({0}), 3.0f, 1e-5f);
}

TEST_F(DatasetIteratorTest, EqualityAndInequality) {
    auto a = ds_.begin();
    auto b = ds_.begin();
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    ++b;
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
}

TEST_F(DatasetIteratorTest, RangeForTraversesAllRows) {
    std::vector<float> col0vals;
    for (auto& row : ds_)
        col0vals.push_back(row.at({0}));
    ASSERT_EQ(col0vals.size(), 3u);
    EXPECT_NEAR(col0vals[0], 1.0f, 1e-5f);
    EXPECT_NEAR(col0vals[1], 3.0f, 1e-5f);
    EXPECT_NEAR(col0vals[2], 5.0f, 1e-5f);
}

// --- DatasetInner: const_iterator ------------------------------------------

class DatasetConstIteratorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        ds_ = make3x2Dataset<f32>("citer_");
    }
    DatasetInner<f32> ds_;

    const DatasetInner<f32>& cds() const { return ds_; }
};

TEST_F(DatasetConstIteratorTest, BeginNotEqualEnd) {
    EXPECT_NE(cds().begin(), cds().end());
}

TEST_F(DatasetConstIteratorTest, DereferenceReturnsFirstRow) {
    const auto& row = *cds().begin();
    EXPECT_NEAR(row.at({0}), 1.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, ArrowReturnsPointer) {
    EXPECT_EQ(cds().begin()->getDimensions(), 1u);
}

TEST_F(DatasetConstIteratorTest, PreIncrement) {
    auto it = cds().begin();
    auto& ret = ++it;
    EXPECT_EQ(&ret, &it);
    EXPECT_NEAR((*it).at({0}), 3.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, PostIncrement) {
    auto it = cds().begin();
    auto old = it++;
    EXPECT_NEAR((*old).at({0}), 1.0f, 1e-5f);
    EXPECT_NEAR((*it).at({0}), 3.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, PreDecrement) {
    auto it = cds().end();
    auto& ret = --it;
    EXPECT_EQ(&ret, &it);
    EXPECT_NEAR((*it).at({0}), 5.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, PostDecrement) {
    auto it = cds().end();
    auto old = it--;
    EXPECT_EQ(old, cds().end());
    EXPECT_NEAR((*it).at({0}), 5.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, PlusEqualsN) {
    auto it = cds().begin();
    it += 2;
    EXPECT_NEAR((*it).at({0}), 5.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, OperatorPlusN) {
    EXPECT_NEAR((*(cds().begin() + 1)).at({0}), 3.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, MinusEqualsN) {
    auto it = cds().end();
    it -= 1;
    EXPECT_NEAR((*it).at({0}), 5.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, OperatorMinusN) {
    EXPECT_NEAR((*(cds().end() - 2)).at({0}), 3.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, IteratorDifference) {
    EXPECT_EQ(cds().end() - cds().begin(), 3);
}

TEST_F(DatasetConstIteratorTest, SubscriptOperator) {
    auto row2 = cds().begin()[2];
    EXPECT_NEAR(row2.at({0}), 5.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, EqualityAndInequality) {
    auto a = cds().begin();
    auto b = cds().begin();
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    ++b;
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
}

TEST_F(DatasetConstIteratorTest, CBeginCEndRangeFor) {
    std::vector<float> col1vals;
    for (auto it = ds_.cbegin(); it != ds_.cend(); ++it)
        col1vals.push_back((*it).at({1}));
    ASSERT_EQ(col1vals.size(), 3u);
    EXPECT_NEAR(col1vals[0], 2.0f, 1e-5f);
    EXPECT_NEAR(col1vals[1], 4.0f, 1e-5f);
    EXPECT_NEAR(col1vals[2], 6.0f, 1e-5f);
}

TEST_F(DatasetConstIteratorTest, ConstRangeForVisitsAllRows) {
    int count = 0;
    for (const auto& row : cds()) {
        EXPECT_EQ(row.getDimensions(), 1u);
        ++count;
    }
    EXPECT_EQ(count, 3);
}

// --- Typed tests: DatasetInner<f32> and DatasetInner<f64> ------------------
// Verifies template instantiation + basic loader behaviour for both float types.

using FloatTypes = ::testing::Types<f32, f64>;

template <typename T>
class DatasetInnerTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE(DatasetInnerTypedTest, FloatTypes);

TYPED_TEST(DatasetInnerTypedTest, LoadAndSizeCorrect) {
    const std::string tag = std::string("typed_") + typeid(TypeParam).name();
    DatasetInner<TypeParam> ds;
    DatasetInnerLoader().loadFromCSVTo<TypeParam>(
        writeTempCSV("x,y\n1.0,2.0\n3.0,4.0\n", tag), ds);
    EXPECT_EQ(ds.size(), 2u);
}

TYPED_TEST(DatasetInnerTypedTest, ValuesAreCorrect) {
    const std::string tag = std::string("typed_val_") + typeid(TypeParam).name();
    DatasetInner<TypeParam> ds;
    DatasetInnerLoader().loadFromCSVTo<TypeParam>(
        writeTempCSV("x,y\n7.0,8.0\n", tag), ds);
    ASSERT_EQ(ds.size(), 1u);
    auto row = ds.getItem(0);
    EXPECT_NEAR(static_cast<double>(row.at({0})), 7.0, 1e-5);
    EXPECT_NEAR(static_cast<double>(row.at({1})), 8.0, 1e-5);
}

TYPED_TEST(DatasetInnerTypedTest, EmptyDataSectionProducesEmptyDataset) {
    const std::string tag = std::string("typed_empty_") + typeid(TypeParam).name();
    DatasetInner<TypeParam> ds;
    DatasetInnerLoader().loadFromCSVTo<TypeParam>(
        writeTempCSV("x,y\n", tag), ds);
    EXPECT_EQ(ds.size(), 0u);
    EXPECT_EQ(ds.begin(), ds.end());
}

TYPED_TEST(DatasetInnerTypedTest, UseZeroBlankStrategy) {
    const std::string tag = std::string("typed_zero_") + typeid(TypeParam).name();
    DatasetInner<TypeParam> ds;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::UseZero);
    loader.loadFromCSVTo<TypeParam>(writeTempCSV("x,y\n5.0,\n2.0,3.0\n", tag), ds);
    ASSERT_EQ(ds.size(), 2u);
    EXPECT_NEAR(static_cast<double>(ds.getItem(0).at({1})), 0.0, 1e-5);
}

TYPED_TEST(DatasetInnerTypedTest, SetNanBlankStrategy) {
    const std::string tag = std::string("typed_nan_") + typeid(TypeParam).name();
    DatasetInner<TypeParam> ds;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::SetNan);
    loader.loadFromCSVTo<TypeParam>(writeTempCSV("x,y\n1.0,\n", tag), ds);
    ASSERT_EQ(ds.size(), 1u);
    EXPECT_TRUE(std::isnan(static_cast<double>(ds.getItem(0).at({1}))));
}

TYPED_TEST(DatasetInnerTypedTest, JumpOneBlankStrategy) {
    const std::string tag = std::string("typed_jump_") + typeid(TypeParam).name();
    DatasetInner<TypeParam> ds;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::JumpOne);
    loader.loadFromCSVTo<TypeParam>(writeTempCSV("x,y\n1.0,\n3.0,4.0\n", tag), ds);
    EXPECT_EQ(ds.size(), 1u);
}

TYPED_TEST(DatasetInnerTypedTest, IteratorDifferenceEqualsSize) {
    const std::string tag = std::string("typed_diff_") + typeid(TypeParam).name();
    DatasetInner<TypeParam> ds;
    DatasetInnerLoader().loadFromCSVTo<TypeParam>(
        writeTempCSV("a,b\n1.0,2.0\n3.0,4.0\n5.0,6.0\n", tag), ds);
    using diff_t = typename DatasetInner<TypeParam>::difference_type;
    EXPECT_EQ(ds.end() - ds.begin(), static_cast<diff_t>(ds.size()));
}
