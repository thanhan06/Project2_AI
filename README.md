## 🏗️ Cấu trúc Project
```
Project2_DecisionTree/
├── 📁 data/                     # Dữ liệu gốc
├── 📁 notebooks/                # Jupyter notebooks phân tích
│   ├── 1_heart_disease.ipynb
│   ├── 2_penguins.ipynb
│   ├── 3_custom_dataset.ipynb
│   └── compare_datasets.ipynb
├── 📁 reports/                  # Báo cáo và hình ảnh
├── requirements.txt             # Dependencies
├── prompts_used.txt             # Lịch sử sử dụng prompt ChatGPT
└── README.md                    # File mô tả dự án (file này)
```

---

# 📊 Project 2 – Decision Tree Classifier

## 👨‍🏫 Course: CS14003 – Introduction to Artificial Intelligence

## 🔍 Tasks Implemented
- [x] Load & visualize 3 datasets
- [x] Stratified train/test split with 4 ratios (40/60, 60/40, 80/20, 90/10)
- [x] Train Decision Tree using Information Gain (entropy)
- [x] Evaluation: classification report + confusion matrix
- [x] Analyze `max_depth` effect on 80/20 split
- [x] Compare datasets in terms of structure & accuracy

---

## ▶️ How to Run
1. **Install dependencies:**
```bash
pip install -r requirements.txt
```
2. **Run individual notebooks in `notebooks/` using Jupyter Notebook**

---

## 📌 Notes
- Cây quyết định hiển thị tối đa depth=3 để dễ quan sát
- Phân tích ảnh hưởng `max_depth` thực hiện trên tập 80/20
- `compare_datasets.ipynb` dùng để tổng hợp kết quả và trực quan hóa so sánh

---

