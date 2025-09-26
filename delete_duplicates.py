import os

def delete_duplicates(folder1, folder2):
    # Lấy danh sách file trong 2 folder
    files1 = set(os.listdir(folder1))   # set để tìm nhanh
    files2 = os.listdir(folder2)

    count = 0
    for f in files2:
        if f in files1:  # nếu file trùng tên + đuôi
            file_path = os.path.join(folder2, f)
            try:
                os.remove(file_path)
                count += 1
                print(f"Đã xóa: {file_path}")
            except Exception as e:
                print(f"Lỗi khi xóa {file_path}: {e}")

    print(f"Tổng số file đã xóa: {count}")

if __name__ == "__main__":
    folder1 = r"C:\Users\hauhm\Downloads\drive-download-20250921T095718Z-1-001\test\labels"  # thay bằng đường dẫn folder gốc
    folder2 = r"C:\Users\hauhm\Downloads\hole_project\hole_project\data_segment\test\labels"  # thay bằng folder chứa data bị add nhầm
    delete_duplicates(folder1, folder2)
