// Copyright (c) $todat.year-2026 Contributors of Hahaha
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

#ifndef HAHAHA_DATASETINNERLOADER_H_A4976991CB27480FA1D8C8FD93AE7B12
#define HAHAHA_DATASETINNERLOADER_H_A4976991CB27480FA1D8C8FD93AE7B12
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#include "DatasetInner.h"
#include "common/errors/Error.h"
#include "math/TensorWrapper.h"
#include "math/ds/TensorShape.h"
#include "ml/dataset/DatasetHandleBlankStrategy.h"
#include "ml/dataset/DatasetTypeUnifyStrategy.h"
#include "ml/dataset/dataset_format.h"
#include "utils/common/HelperStruct.h"
#include "utils/common/StringUtils.h"
#include "utils/log/Logger.h"

namespace hahaha::ml {

class DatasetInnerLoader {
    static constexpr auto CSVLineDelimiter = ',';

  public:
    template <typename T>
    void loadFromCSVTo(const std::string& pathString,
                       DatasetInner<T>& dataset) {
        currFile_ = pathString;
        currLine_ = 1;
        checkPathExist(pathString);
        std::ifstream ifs(pathString);
        if (!ifs.is_open()) {
            error(std::format("file can not open: {}", pathString));
            throw std::runtime_error(pathString);
        }

        setUpColumnNames<T>(ifs, dataset);
        setUpData<T>(ifs, dataset);
        dataset.datasetName_ = pathString;
    }

    template <typename T>
    void loadFromCSVTo(const std::string& pathString,
                       DatasetInner<T>& dataset,
                       CSVDatasetFormat format) {
        currFile_ = pathString;
        currLine_ = 1;
        checkPathExist(pathString);
        std::ifstream ifs(pathString);
        if (!ifs.is_open()) {
            error(std::format("file can not open: {}", pathString));
            throw std::runtime_error(pathString);
        }

        setUpColumnNames<T>(ifs, dataset);
        setUpData<T>(ifs, dataset);
        dataset.datasetName_ = pathString;
    }

    static void checkPathExist(const std::string& pathString) {
        if (!std::filesystem::exists(pathString)) {
            error(std::format("no files are founded: {}", pathString));
            throw std::invalid_argument("pathString is not exist");
        }
    }

    void setHandleBlankStrategy(DatasetHandleBlankStrategy strategy) {
        datasetHandleBlankStrategy_ = strategy;
    }

  private:
    template <typename T>
    std::expected<void, common::Error> setUpData(std::ifstream& ifs,
                                                 DatasetInner<T>& dataset) {
        auto line = std::string{};

        std::vector<std::vector<T>> dataList;
        while (std::getline(ifs, line)) {
            auto datas = handleOneLine<T>(line);
            if (!datas.has_value() || datas.value().size() == 0) {
                continue;
            }
            dataList.push_back(datas.value());
        }

        fillData(dataList, dataset);
        return {};
    }

    template <typename T>
    std::expected<std::vector<T>, common::Error>
    handleOneLine(const std::string& line) {
        const auto strVec =
            utils::StringUtils::split(line, CSVLineDelimiter, true);
        auto res = std::vector<T>{};

        if (strVec.size() != columnNum_) {
            error(std::format("error: when parsing the line:{} at file {}:{}",
                              line,
                              currFile_,
                              static_cast<int>(currLine_)));
            return std::unexpected(common::InvalidDatasetError());
        }

        res.resize(strVec.size());

        for (size_t i = 0; i < strVec.size(); ++i) {
            auto valRes = handleOneValue<T>(strVec[i]);
            if (!valRes) {
                error(
                    std::format("error: when parsing the line:{} at file {}:{}",
                                line,
                                currFile_,
                                static_cast<int>(currLine_)));
                return std::unexpected(common::InvalidDatasetError());
            }
            res[i] = valRes.value();
        }

        return res;
    }

    template <typename T>
    std::expected<T, common::Error> handleOneValue(const std::string& str) {
        if (utils::StringUtils::isBlank(str)) {
            switch (datasetHandleBlankStrategy_) {
            case DatasetHandleBlankStrategy::SetNan:
                if constexpr (utils::isLegalFloatType<T>::value) {
                    return static_cast<T>(std::numeric_limits<T>::quiet_NaN());
                } else {
                    error("Invalid dataset value type for SetNan strategy");
                    return std::unexpected(common::InvalidDatasetError());
                }
                break;

            case DatasetHandleBlankStrategy::UseZero:
                return static_cast<T>(0);
                break;

            default:
                error("Invalid dataset handle blank strategy");
                return std::unexpected(common::InvalidDatasetError());
            }
        }
        return utils::StringUtils::to<T>(str);
    }

    template <typename T>
    void setUpColumnNames(std::ifstream& ifs, DatasetInner<T>& dataset) {
        auto line = std::string{};
        std::getline(ifs, line);
        auto columns = utils::StringUtils::split(line, CSVLineDelimiter);
        columnNum_ = columns.size();
        dataset.features_ = std::move(columns);
    }

    template <typename T>
    void fillData(std::vector<std::vector<T>>& dataList,
                  DatasetInner<T>& dataset) {
        if (dataList.empty()) {
            dataset.sample = Tensor<T>();
            return;
        }

        Tensor<T> sampleTensor =
            Tensor<T>::buildFromShape({dataList.size(), dataList[0].size()});

        for (size_t i = 0; i < dataList.size(); ++i) {
            for (size_t j = 0; j < dataList[i].size(); ++j) {
                sampleTensor.getComputeNode()
                    ->getData()
                    ->getRawData()[i * dataList[0].size() + j] = dataList[i][j];
            }
        }

        dataset.sample = sampleTensor;
    }

    void clearStatus() {
        columnNum_ = 0;
        currLine_ = 1;
        currFile_ = "";
    }

    size_t columnNum_ = 0;
    std::string currFile_;
    size_t currLine_ = 0;

    DatasetHandleBlankStrategy datasetHandleBlankStrategy_ =
        getDefaultDatasetHandleBlankStrategy();
};

} // namespace hahaha::ml

#endif // HAHAHA_DATASETINNERLOADER_H_A4976991CB27480FA1D8C8FD93AE7B12
