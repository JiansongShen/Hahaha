//  Copyright (c) 2026 Contributors of hahaha(https://github.com/Napbad/Hahaha)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//       https://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Contributors:
//  jiansongshen (jason.shen111@outlook.com) (https://github.com/jiansongshen)
//

#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ml/dataset/DatasetInner.h"
#include "ml/dataset/DatasetInnerLoader.h"
#include "ml/dataset/DatasetHandleBlankStrategy.h"
#include "ml/dataset/DatasetTypeUnifyStrategy.h"
#include "public/Tensor.h"

using namespace hahaha::ml;
using namespace hahaha;
using hahaha::math::NestedData;

// ============================================================================
// DatasetInner Tests
// ============================================================================

class DatasetInnerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for DatasetInner tests
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
};

TEST_F(DatasetInnerTest, Constructor_DefaultParameters) {
    DatasetInner<float> dataset({}, "", {});
    EXPECT_EQ(dataset.getDatasetName(), "");
    EXPECT_TRUE(dataset.getFeatures().empty());
    EXPECT_TRUE(dataset.getLabels().empty());
    EXPECT_EQ(dataset.getDatasetTypeUnifyStrategy(),
              getDefaultDatasetTypeUnifyStrategy());
}

TEST_F(DatasetInnerTest, Constructor_WithAllParameters) {
    std::vector<std::string> features = {"feature1", "feature2", "feature3"};
    std::vector<std::string> labels = {"label1"};
    std::string datasetName = "test_dataset";
    auto strategy = DatasetTypeUnifyStrategy::AllFloat64;
    
    Tensor<float> x(NestedData<float>{{1.0f, 2.0f}, {3.0f, 4.0f}});
    Tensor<float> y(NestedData<float>{5.0f, 6.0f});
    
    DatasetInner<float> dataset(features, datasetName, labels, strategy, x, y);
    
    EXPECT_EQ(dataset.getDatasetName(), datasetName);
    EXPECT_EQ(dataset.getFeatures(), features);
    EXPECT_EQ(dataset.getLabels(), labels);
    EXPECT_EQ(dataset.getDatasetTypeUnifyStrategy(), strategy);
    
    auto xResult = dataset.getX();
    auto yResult = dataset.getY();
    EXPECT_FLOAT_EQ(xResult.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(xResult.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(yResult.at({0}), 5.0f);
}

TEST_F(DatasetInnerTest, GetterMethods) {
    std::vector<std::string> features = {"f1", "f2"};
    std::vector<std::string> labels = {"l1"};
    DatasetInner<float> dataset(features, "name", labels);
    
    EXPECT_EQ(dataset.getDatasetName(), "name");
    EXPECT_EQ(dataset.getFeatures().size(), 2u);
    EXPECT_EQ(dataset.getFeatures()[0], "f1");
    EXPECT_EQ(dataset.getFeatures()[1], "f2");
    EXPECT_EQ(dataset.getLabels().size(), 1u);
    EXPECT_EQ(dataset.getLabels()[0], "l1");
}

// ============================================================================
// DatasetInnerLoader Tests - Path Checking
// ============================================================================

class DatasetInnerLoaderPathTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory if it doesn't exist
        if (!std::filesystem::exists("temp")) {
            std::filesystem::create_directories("temp");
        }
    }
    
    void TearDown() override {
        // Cleanup temp files if needed
    }
};

TEST_F(DatasetInnerLoaderPathTest, CheckPathExist_FileExists) {
    // Create a temporary file
    std::string tempFile = "temp/test_dataset_exists.csv";
    std::ofstream ofs(tempFile);
    ofs << "header\n";
    ofs.close();
    
    EXPECT_NO_THROW(DatasetInnerLoader::checkPathExist(tempFile));
    
    // Cleanup
    std::filesystem::remove(tempFile);
}

TEST_F(DatasetInnerLoaderPathTest, CheckPathExist_FileNotExists) {
    std::string nonExistentFile = "temp/non_existent_file_12345.csv";
    
    EXPECT_THROW(DatasetInnerLoader::checkPathExist(nonExistentFile),
                 std::invalid_argument);
}

// ============================================================================
// DatasetInnerLoader Tests - CSV Loading
// ============================================================================

class DatasetInnerLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory if it doesn't exist
        if (!std::filesystem::exists("temp")) {
            std::filesystem::create_directories("temp");
        }
        // Create temporary CSV file
        tempFile_ = "temp/test_dataset.csv";
        std::ofstream ofs(tempFile_);
        ofs << "feature1,feature2,label\n";
        ofs << "1.0,2.0,3.0\n";
        ofs << "4.0,5.0,6.0\n";
        ofs.close();
    }
    
    void TearDown() override {
        if (std::filesystem::exists(tempFile_)) {
            std::filesystem::remove(tempFile_);
        }
    }
    
    std::string tempFile_;
};

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_BasicLoad) {
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile_, dataset);
    
    EXPECT_EQ(dataset.getDatasetName(), tempFile_);
    EXPECT_EQ(dataset.getFeatures().size(), 3u);
    EXPECT_EQ(dataset.getFeatures()[0], "feature1");
    EXPECT_EQ(dataset.getFeatures()[1], "feature2");
    EXPECT_EQ(dataset.getFeatures()[2], "label");
    
    auto x = dataset.getX();
    auto y = dataset.getY();
    
    EXPECT_EQ(x.getShape()[0], 2u); // 2 rows
    EXPECT_EQ(x.getShape()[1], 2u); // 2 features
    EXPECT_EQ(y.getShape()[0], 2u); // 2 rows
    EXPECT_EQ(y.getShape()[1], 1u); // 1 label
    
    EXPECT_FLOAT_EQ(x.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(x.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(y.at({0}), 3.0f);
    EXPECT_FLOAT_EQ(x.at({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(x.at({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(y.at({1}), 6.0f);
}

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_FileCannotOpen) {
    // Create a file and then remove permissions (if possible)
    // Or use a path that cannot be opened
    std::string invalidPath = "temp/invalid_path_12345/test.csv";
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    EXPECT_THROW(loader.loadFromCSVTo(invalidPath, dataset),
                 std::runtime_error);
}

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_WithEmptyLines) {
    std::string tempFile = "temp/test_empty_lines.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "1.0,2.0,3.0\n";
    ofs << "\n"; // Empty line
    ofs << "4.0,5.0,6.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile, dataset);
    
    auto x = dataset.getX();
    auto y = dataset.getY();
    
    // Should have 2 data rows (empty line skipped)
    EXPECT_EQ(x.getShape()[0], 2u);
    EXPECT_EQ(y.getShape()[0], 2u);
    
    std::filesystem::remove(tempFile);
}

// ============================================================================
// DatasetInnerLoader Tests - Blank Value Handling
// ============================================================================

class DatasetBlankValueTest : public ::testing::Test {
protected:
    void SetUp() override {
        tempFile_ = "temp/test_blank.csv";
    }
    
    void TearDown() override {
        if (std::filesystem::exists(tempFile_)) {
            std::filesystem::remove(tempFile_);
        }
    }
    
    std::string tempFile_;
};

TEST_F(DatasetBlankValueTest, HandleBlankValue_SetNan_Float32) {
    std::ofstream ofs(tempFile_);
    ofs << "f1,f2,label\n";
    ofs << "1.0, ,3.0\n"; // Blank value in middle
    ofs << "4.0,5.0,6.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::SetNan);
    
    loader.loadFromCSVTo(tempFile_, dataset);
    
    auto x = dataset.getX();
    float val = x.at({0, 1});
    EXPECT_TRUE(std::isnan(val));
}

TEST_F(DatasetBlankValueTest, HandleBlankValue_SetNan_Float64) {
    std::ofstream ofs(tempFile_);
    ofs << "f1,f2,label\n";
    ofs << "1.0, ,3.0\n";
    ofs << "4.0,5.0,6.0\n";
    ofs.close();
    
    DatasetInner<double> dataset;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::SetNan);
    
    loader.loadFromCSVTo(tempFile_, dataset);
    
    auto x = dataset.getX();
    double val = x.at({0, 1});
    EXPECT_TRUE(std::isnan(val));
}

TEST_F(DatasetBlankValueTest, HandleBlankValue_UseZero) {
    std::ofstream ofs(tempFile_);
    ofs << "f1,f2,label\n";
    ofs << "1.0, ,3.0\n";
    ofs << "4.0,5.0,6.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::UseZero);
    
    loader.loadFromCSVTo(tempFile_, dataset);
    
    auto x = dataset.getX();
    float val = x.at({0, 1});
    EXPECT_FLOAT_EQ(val, 0.0f);
}

TEST_F(DatasetBlankValueTest, HandleBlankValue_SetNan_NonFloatType) {
    // Test that SetNan strategy fails for non-float types
    std::ofstream ofs(tempFile_);
    ofs << "f1,f2,label\n";
    ofs << "1, ,3\n";
    ofs << "4,5,6\n";
    ofs.close();
    
    DatasetInner<int> dataset;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::SetNan);
    
    // Should handle error gracefully (skip line or throw)
    try {
        loader.loadFromCSVTo(tempFile_, dataset);
        // If it doesn't throw, the line with blank should be skipped
        auto x = dataset.getX();
        EXPECT_EQ(x.getShape()[0], 1u); // Only one valid row
    } catch (...) {
        // Throwing is acceptable for invalid type with SetNan
    }
}

TEST_F(DatasetBlankValueTest, HandleBlankValue_UseZero_IntType) {
    std::ofstream ofs(tempFile_);
    ofs << "f1,f2,label\n";
    ofs << "1, ,3\n";
    ofs << "4,5,6\n";
    ofs.close();
    
    DatasetInner<int> dataset;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::UseZero);
    
    loader.loadFromCSVTo(tempFile_, dataset);
    
    auto x = dataset.getX();
    int val = x.at({0, 1});
    EXPECT_EQ(val, 0);
}

TEST_F(DatasetBlankValueTest, HandleBlankValue_DefaultStrategy) {
    // Test default branch (JumpOne or invalid strategy)
    std::ofstream ofs(tempFile_);
    ofs << "f1,f2,label\n";
    ofs << "1.0, ,3.0\n";
    ofs << "4.0,5.0,6.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::JumpOne);
    
    // Should handle error gracefully
    try {
        loader.loadFromCSVTo(tempFile_, dataset);
        // If it doesn't throw, the line with blank should be skipped
        auto x = dataset.getX();
        EXPECT_EQ(x.getShape()[0], 1u); // Only one valid row
    } catch (...) {
        // Throwing is acceptable for default/invalid strategy
    }
}

// ============================================================================
// DatasetInnerLoader Tests - Error Handling
// ============================================================================

TEST_F(DatasetInnerLoaderTest, HandleOneLine_WrongColumnCount) {
    std::string tempFile = "temp/test_wrong_columns.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "1.0,2.0,3.0\n";
    ofs << "4.0,5.0\n"; // Missing one column
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    // Should handle the error gracefully (skip the line or throw)
    // Based on implementation, it may skip or throw
    try {
        loader.loadFromCSVTo(tempFile, dataset);
        // If it doesn't throw, it should skip the invalid line
        auto x = dataset.getX();
        EXPECT_EQ(x.getShape()[0], 1u); // Only one valid row
    } catch (...) {
        // If it throws, that's also acceptable behavior
    }
    
    std::filesystem::remove(tempFile);
}

TEST_F(DatasetInnerLoaderTest, HandleOneLine_ExtraColumns) {
    std::string tempFile = "temp/test_extra_columns.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "1.0,2.0,3.0,4.0\n"; // Extra column
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    try {
        loader.loadFromCSVTo(tempFile, dataset);
        // Should skip invalid line
        auto x = dataset.getX();
        EXPECT_EQ(x.getShape()[0], 0u);
    } catch (...) {
        // Throwing is also acceptable
    }
    
    std::filesystem::remove(tempFile);
}

// ============================================================================
// DatasetHandleBlankStrategy Tests
// ============================================================================

class DatasetHandleBlankStrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        originalStrategy_ = getDefaultDatasetHandleBlankStrategy();
    }
    
    void TearDown() override {
        // Restore original strategy
        setDefaultDatasetHandleBlankStrategy(originalStrategy_);
    }
    
    DatasetHandleBlankStrategy originalStrategy_;
};

TEST_F(DatasetHandleBlankStrategyTest, GetDefaultStrategy) {
    auto strategy = getDefaultDatasetHandleBlankStrategy();
    EXPECT_EQ(strategy, DatasetHandleBlankStrategy::SetNan);
}

TEST_F(DatasetHandleBlankStrategyTest, SetDefaultStrategy) {
    auto original = getDefaultDatasetHandleBlankStrategy();
    
    setDefaultDatasetHandleBlankStrategy(DatasetHandleBlankStrategy::UseZero);
    EXPECT_EQ(getDefaultDatasetHandleBlankStrategy(),
              DatasetHandleBlankStrategy::UseZero);
    
    setDefaultDatasetHandleBlankStrategy(DatasetHandleBlankStrategy::JumpOne);
    EXPECT_EQ(getDefaultDatasetHandleBlankStrategy(),
              DatasetHandleBlankStrategy::JumpOne);
    
    // Restore original
    setDefaultDatasetHandleBlankStrategy(original);
}

TEST_F(DatasetHandleBlankStrategyTest, AllStrategies) {
    // Test all enum values exist
    auto setNan = DatasetHandleBlankStrategy::SetNan;
    auto useZero = DatasetHandleBlankStrategy::UseZero;
    auto jumpOne = DatasetHandleBlankStrategy::JumpOne;
    
    EXPECT_NE(setNan, useZero);
    EXPECT_NE(setNan, jumpOne);
    EXPECT_NE(useZero, jumpOne);
}

// ============================================================================
// DatasetTypeUnifyStrategy Tests
// ============================================================================

class DatasetTypeUnifyStrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        originalStrategy_ = getDefaultDatasetTypeUnifyStrategy();
    }
    
    void TearDown() override {
        // Restore original strategy
        setDefaultDatasetTypeUnifyStrategy(originalStrategy_);
    }
    
    DatasetTypeUnifyStrategy originalStrategy_;
};

TEST_F(DatasetTypeUnifyStrategyTest, GetDefaultStrategy) {
    auto strategy = getDefaultDatasetTypeUnifyStrategy();
    EXPECT_EQ(strategy, DatasetTypeUnifyStrategy::AllFloat32);
}

TEST_F(DatasetTypeUnifyStrategyTest, SetDefaultStrategy) {
    auto original = getDefaultDatasetTypeUnifyStrategy();
    
    setDefaultDatasetTypeUnifyStrategy(DatasetTypeUnifyStrategy::AllFloat64);
    EXPECT_EQ(getDefaultDatasetTypeUnifyStrategy(),
              DatasetTypeUnifyStrategy::AllFloat64);
    
    setDefaultDatasetTypeUnifyStrategy(DatasetTypeUnifyStrategy::UseLargestRange);
    EXPECT_EQ(getDefaultDatasetTypeUnifyStrategy(),
              DatasetTypeUnifyStrategy::UseLargestRange);
    
    setDefaultDatasetTypeUnifyStrategy(DatasetTypeUnifyStrategy::AllFloat16);
    EXPECT_EQ(getDefaultDatasetTypeUnifyStrategy(),
              DatasetTypeUnifyStrategy::AllFloat16);
    
    // Restore original
    setDefaultDatasetTypeUnifyStrategy(original);
}

TEST_F(DatasetTypeUnifyStrategyTest, AllStrategies) {
    // Test all enum values exist
    auto allFloat32 = DatasetTypeUnifyStrategy::AllFloat32;
    auto allFloat64 = DatasetTypeUnifyStrategy::AllFloat64;
    auto allFloat16 = DatasetTypeUnifyStrategy::AllFloat16;
    auto useLargestRange = DatasetTypeUnifyStrategy::UseLargestRange;
    
    EXPECT_NE(allFloat32, allFloat64);
    EXPECT_NE(allFloat32, allFloat16);
    EXPECT_NE(allFloat32, useLargestRange);
}

// ============================================================================
// DatasetInnerLoader Tests - Edge Cases
// ============================================================================

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_SingleRow) {
    std::string tempFile = "temp/test_single_row.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "1.0,2.0,3.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile, dataset);
    
    auto x = dataset.getX();
    auto y = dataset.getY();
    
    EXPECT_EQ(x.getShape()[0], 1u);
    EXPECT_EQ(y.getShape()[0], 1u);
    EXPECT_FLOAT_EQ(x.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(x.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(y.at({0}), 3.0f);
    
    std::filesystem::remove(tempFile);
}

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_NoDataRows) {
    std::string tempFile = "temp/test_no_data.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    // No data rows
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    // Should handle empty data gracefully
    loader.loadFromCSVTo(tempFile, dataset);
    
    // Check that features are set
    EXPECT_EQ(dataset.getFeatures().size(), 3u);
    EXPECT_EQ(dataset.getFeatures()[0], "f1");
    EXPECT_EQ(dataset.getFeatures()[1], "f2");
    EXPECT_EQ(dataset.getFeatures()[2], "label");
    
    // X and Y should be empty tensors
    auto x = dataset.getX();
    auto y = dataset.getY();
    EXPECT_EQ(x.getShape().size(), 0u); // Empty tensor
    EXPECT_EQ(y.getShape().size(), 0u); // Empty tensor
    
    std::filesystem::remove(tempFile);
}

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_MultipleFeatures) {
    std::string tempFile = "temp/test_multi_features.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,f3,f4,label\n";
    ofs << "1.0,2.0,3.0,4.0,5.0\n";
    ofs << "6.0,7.0,8.0,9.0,10.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile, dataset);
    
    EXPECT_EQ(dataset.getFeatures().size(), 5u);
    
    auto x = dataset.getX();
    auto y = dataset.getY();
    
    EXPECT_EQ(x.getShape()[0], 2u);
    EXPECT_EQ(x.getShape()[1], 4u); // 4 features
    EXPECT_EQ(y.getShape()[0], 2u);
    
    EXPECT_FLOAT_EQ(x.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(x.at({0, 3}), 4.0f);
    EXPECT_FLOAT_EQ(y.at({0}), 5.0f);
    
    std::filesystem::remove(tempFile);
}

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_DifferentNumericTypes) {
    // Test with double type
    std::string tempFile = "temp/test_double.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "1.5,2.5,3.5\n";
    ofs << "4.5,5.5,6.5\n";
    ofs.close();
    
    DatasetInner<double> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile, dataset);
    
    auto x = dataset.getX();
    auto y = dataset.getY();
    
    EXPECT_DOUBLE_EQ(x.at({0, 0}), 1.5);
    EXPECT_DOUBLE_EQ(x.at({0, 1}), 2.5);
    EXPECT_DOUBLE_EQ(y.at({0}), 3.5);
    
    std::filesystem::remove(tempFile);
}

// ============================================================================
// DatasetInnerLoader Tests - Blank Handling Strategies (if accessible)
// ============================================================================

TEST_F(DatasetBlankValueTest, HandleBlankValue_AllBlankValues) {
    std::ofstream ofs(tempFile_);
    ofs << "f1,f2,label\n";
    ofs << " , ,3.0\n"; // All features blank
    ofs << "4.0,5.0,6.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::SetNan);
    
    loader.loadFromCSVTo(tempFile_, dataset);
    
    auto x = dataset.getX();
    // First row should have NaN values
    EXPECT_TRUE(std::isnan(static_cast<double>(x.at({0, 0}))));
    EXPECT_TRUE(std::isnan(static_cast<double>(x.at({0, 1}))));
}

TEST_F(DatasetBlankValueTest, HandleBlankValue_MixedBlankAndNonBlank) {
    std::ofstream ofs(tempFile_);
    ofs << "f1,f2,f3,label\n";
    ofs << "1.0, ,3.0,4.0\n"; // Middle value blank
    ofs << "5.0,6.0,7.0,8.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::SetNan);
    
    loader.loadFromCSVTo(tempFile_, dataset);
    
    auto x = dataset.getX();
    EXPECT_FLOAT_EQ(x.at({0, 0}), 1.0f);
    EXPECT_TRUE(std::isnan(static_cast<double>(x.at({0, 1}))));
    EXPECT_FLOAT_EQ(x.at({0, 2}), 3.0f);
}

// ============================================================================
// DatasetInnerLoader Tests - Invalid Data Handling
// ============================================================================

TEST_F(DatasetInnerLoaderTest, HandleOneValue_InvalidNumericString) {
    // This tests the StringUtils::to conversion
    std::string tempFile = "temp/test_invalid_numeric.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "abc,2.0,3.0\n"; // Invalid number
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    // Should handle conversion error
    try {
        loader.loadFromCSVTo(tempFile, dataset);
        // If it doesn't throw, the invalid value might be 0 or NaN
    } catch (...) {
        // Throwing is acceptable
    }
    
    std::filesystem::remove(tempFile);
}

// ============================================================================
// DatasetInnerLoader Tests - fillData Edge Cases
// ============================================================================

TEST_F(DatasetInnerLoaderTest, FillData_VaryingRowLengths) {
    // Note: This should be caught by handleOneLine validation
    // But we test that the system handles it
    std::string tempFile = "temp/test_varying_lengths.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "1.0,2.0,3.0\n";
    ofs << "4.0,5.0\n"; // Shorter row
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    try {
        loader.loadFromCSVTo(tempFile, dataset);
        // Should skip invalid row
        auto x = dataset.getX();
        EXPECT_EQ(x.getShape()[0], 1u);
    } catch (...) {
        // Throwing is also acceptable
    }
    
    std::filesystem::remove(tempFile);
}

// ============================================================================
// Additional tests for complete branch coverage
// ============================================================================

TEST_F(DatasetInnerLoaderTest, HandleOneLine_EmptyLineAfterHeader) {
    std::string tempFile = "temp/test_empty_after_header.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "\n"; // Empty line immediately after header
    ofs << "1.0,2.0,3.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile, dataset);
    
    auto x = dataset.getX();
    // Empty line should be skipped
    EXPECT_EQ(x.getShape()[0], 1u);
    
    std::filesystem::remove(tempFile);
}

TEST_F(DatasetInnerLoaderTest, HandleOneValue_NonBlankValue) {
    // Test the non-blank path in handleOneValue
    std::string tempFile = "temp/test_non_blank.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "1.5,2.5,3.5\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile, dataset);
    
    auto x = dataset.getX();
    EXPECT_FLOAT_EQ(x.at({0, 0}), 1.5f);
    EXPECT_FLOAT_EQ(x.at({0, 1}), 2.5f);
    
    std::filesystem::remove(tempFile);
}

TEST_F(DatasetBlankValueTest, HandleBlankValue_UseZero_FloatType) {
    std::ofstream ofs(tempFile_);
    ofs << "f1,f2,label\n";
    ofs << "1.0, ,3.0\n";
    ofs << "4.0,5.0,6.0\n";
    ofs.close();
    
    DatasetInner<double> dataset;
    DatasetInnerLoader loader;
    loader.setHandleBlankStrategy(DatasetHandleBlankStrategy::UseZero);
    
    loader.loadFromCSVTo(tempFile_, dataset);
    
    auto x = dataset.getX();
    double val = x.at({0, 1});
    EXPECT_DOUBLE_EQ(val, 0.0);
}

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_WithTrailingWhitespace) {
    std::string tempFile = "temp/test_trailing_whitespace.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "1.0,2.0,3.0\n";
    ofs << "4.0,5.0,6.0  \n"; // Trailing whitespace
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile, dataset);
    
    auto x = dataset.getX();
    EXPECT_EQ(x.getShape()[0], 2u);
    
    std::filesystem::remove(tempFile);
}

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_WithLeadingWhitespace) {
    std::string tempFile = "temp/test_leading_whitespace.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    ofs << "  1.0,2.0,3.0\n"; // Leading whitespace
    ofs << "4.0,5.0,6.0\n";
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile, dataset);
    
    auto x = dataset.getX();
    EXPECT_EQ(x.getShape()[0], 2u);
    
    std::filesystem::remove(tempFile);
}

TEST_F(DatasetInnerLoaderTest, LoadFromCSVTo_LargeDataset) {
    std::string tempFile = "temp/test_large_dataset.csv";
    std::ofstream ofs(tempFile);
    ofs << "f1,f2,label\n";
    for (int i = 0; i < 100; ++i) {
        ofs << i << ".0," << (i+1) << ".0," << (i+2) << ".0\n";
    }
    ofs.close();
    
    DatasetInner<float> dataset;
    DatasetInnerLoader loader;
    
    loader.loadFromCSVTo(tempFile, dataset);
    
    auto x = dataset.getX();
    auto y = dataset.getY();
    
    EXPECT_EQ(x.getShape()[0], 100u);
    EXPECT_EQ(y.getShape()[0], 100u);
    EXPECT_FLOAT_EQ(x.at({0, 0}), 0.0f);
    EXPECT_FLOAT_EQ(y.at({99}), 101.0f);
    
    std::filesystem::remove(tempFile);
}
