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
#include <string>
#include <vector>

#include "DatasetInner.h"
#include "common/errors/Error.h"
#include "ml/dataset/DatasetHandleBlankStrategy.h"
#include "ml/dataset/DatasetTypeUnifyStrategy.h"
#include "utils/common/StringUtils.h"
#include "utils/log/Logger.h"

namespace hahaha::ml {

class DatasetInnerLoader {
    static constexpr auto CSVLineDelimiter = ',';

  public:
    template <typename T>
    void loadFromCSVTo(const std::string& pathString, DatasetInner<T>& dataset) {
        currFile_ = pathString;
        currLine_ = 1;
        checkPathExist(pathString);
        std::ifstream ifs(pathString);
        if (!ifs.is_open()) {
            error(std::format("file can not open: {}", pathString));
            throw std::runtime_error(pathString);
        }

        auto line = std::string{};
        std::getline(ifs, line);
        auto features = utils::StringUtils::split(line, CSVLineDelimiter);
        dataset.features_ = features;
        currLine_ = 1;

        std::vector<std::vector<T>> dataList;

        while (std::getline(ifs, line)) {
            auto datas = handleOneLine<T>(line);
            if (datas.size() == 0) {
                continue;
            }
            dataList.push_back(datas);
        }

        Tensor<T> xTensor({dataList.size(), dataList[0].size() - 1});
        Tensor<T> yTensor({dataList.size(), 1});

        for (size_t i = 0; i < dataList.size(); ++i) {
            for (size_t j = 0; j < dataList[i].size() - 1; ++j) {

                xTensor.getComputeNode()
                    ->getData()
                    ->getRawData()[i * dataList[0].size() + j] = dataList[i][j];
            }

            yTensor.getComputeNode()
                ->getData()
                ->getRawData()[dataList[0].size()] =
                dataList[i][dataList[i].size() - 1];
        }

        dataset.datasetName_ = pathString;
        dataset.x = xTensor;
        dataset.y = yTensor;
    }

    static void checkPathExist(const std::string& pathString) {
        if (!std::filesystem::exists(pathString)) {
            error(std::format("no files are founded: {}", pathString));
            throw std::invalid_argument("pathString is not exist");
        }
    }

  private:
    template <typename T>
    std::expected<std::vector<T>, common::Error>
    handleOneLine(const std::string& line) {
        const auto strVec =
            utils::StringUtils::split(line, CSVLineDelimiter, true);
        auto res = std::vector<T>{};
        if (strVec.size() != featureNum_) {
            error(std::format("error: when parsing the line:{} at file {}:{}",
                              line,
                              currFile_,
                              static_cast<int>(currLine_)));
            return std::unexpected(common::InvalidDatasetError());
        }

        res.resize(strVec.size());

        for (size_t i = 0; i < strVec.size(); ++i) {
            res[i] = utils::StringUtils::to<T>(strVec[i]);
        }

        return res;
    }
    void clearStatus() {
        featureNum_ = 0;
        currLine_ = 1;
        currFile_ = "";
    }

    size_t featureNum_ = 0;
    std::string currFile_ = "";
    size_t currLine_ = 0;

    DatasetTypeUnifyStrategy datasetTypeUnifyStrategy_ =
        getDefaultDatasetTypeUnifyStrategy();

    DatasetHandleBlankStrategy datasetHandleBlankStrategy_ =
        getDefaultDatasetHandleBlankStrategy();
};

} // namespace hahaha::ml

#endif // HAHAHA_DATASETINNERLOADER_H_A4976991CB27480FA1D8C8FD93AE7B12
