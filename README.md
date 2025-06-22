# Project 2: Decision Tree Classification
**CS14003 - Introduction to Artificial Intelligence**

## 📖 Mô tả Project
Dự án này thực hiện phân tích và xây dựng cây quyết định (Decision Tree) trên 3 bộ dữ liệu khác nhau:
- **Heart Disease Dataset**: Dự đoán bệnh tim (binary classification)
- **Palmer Penguins Dataset**: Phân loại loài chim cánh cụt (multi-class classification)
- **Custom Dataset**: Bộ dữ liệu tự chọn

## 🏗️ Cấu trúc Project
```
Project2_DecisionTree/
├── 📁 data/                     # Dữ liệu gốc
├── 📁 notebooks/                # Jupyter notebooks phân tích
├── 📁 reports/                  # Báo cáo và hình ảnh
├── 📁 utils/                    # Hàm tiện ích
├── requirements.txt             # Dependencies
├── README.md                    # File này
└── run_all.ipynb              # Chạy toàn bộ pipeline
```

## 🚀 Hướng dẫn chạy

### 1. Cài đặt môi trường
```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### 2. Chạy analysis
```bash
# Khởi động Jupyter Notebook
jupyter notebook

# Chạy từng notebook theo thứ tự:
# 1. notebooks/1_heart_disease.ipynb
# 2. notebooks/2_penguins.ipynb  
# 3. notebooks/3_custom_dataset.ipynb
# 4. notebooks/4_compare_analysis.ipynb

# Hoặc chạy toàn bộ:
# Mở run_all.ipynb và chạy tất cả cells
```

## 📊 Datasets

### Heart Disease Dataset
- **Nguồn**: UCI Machine Learning Repository
- **Samples**: 303
- **Features**: 13 (age, sex, chest pain, blood pressure, cholesterol, etc.)
- **Target**: Binary (0: No heart disease, 1: Heart disease)

### Palmer Penguins Dataset
- **Nguồn**: Palmer Station Antarctica LTER
- **Samples**: 344
- **Features**: 7 (species, island, bill measurements, flipper length, body mass, sex)
- **Target**: Multi-class (Adelie, Chinstrap, Gentoo)

### Custom Dataset
- **Nguồn**: [Sẽ được cập nhật]
- **Samples**: ≥300
- **Features**: [Sẽ được cập nhật]
- **Target**: [Sẽ được cập nhật]

## 🔧 Các task thực hiện

### 2.1 Data Preparation
- Chia dataset theo tỷ lệ 40/60, 60/40, 80/20, 90/10
- Stratified sampling
- Visualize class distribution

### 2.2 Build Decision Tree Classifiers
- Sử dụng `DecisionTreeClassifier` với information gain
- Visualize decision trees với Graphviz
- Thực hiện với tất cả tỷ lệ train/test

### 2.3 Evaluate Classifiers
- Classification report
- Confusion matrix
- Performance insights

### 2.4 Depth vs Accuracy Analysis
- Test với max_depth: None, 2, 3, 4, 5, 6, 7
- So sánh accuracy trên test set
- Visualize và phân tích

### 2.5 Comparative Analysis
- So sánh performance giữa 3 datasets
- Phân tích ảnh hưởng của số classes, features, samples

## 📝 Báo cáo
- Báo cáo PDF hoàn chỉnh trong thư mục `reports/`
- Bao gồm visualizations, insights, và comparative analysis
- Format chuẩn academic

## 👥 Thành viên nhóm
| STT | Họ tên | MSSV | Công việc được giao | Hoàn thành |
|-----|--------|------|-------------------|------------|
| 1   | [Tên]  | [ID] | [Tasks]           | [%]        |
| 2   | [Tên]  | [ID] | [Tasks]           | [%]        |
| 3   | [Tên]  | [ID] | [Tasks]           | [%]        |
| 4   | [Tên]  | [ID] | [Tasks]           | [%]        |

## 🤖 AI Tools Usage
Nếu có sử dụng AI tools (ChatGPT, etc.), các prompts đã được khai báo trong file `prompts_used.txt`

## 📚 References
- UCI Machine Learning Repository
- Palmer Penguins Dataset (Horst et al., 2020)
- Scikit-learn Documentation
- [Thêm references khác nếu có]

---
*Dự án được thực hiện cho môn CS14003 - Introduction to Artificial Intelligence*