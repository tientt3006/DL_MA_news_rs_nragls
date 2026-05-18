**Tóm tắt bài báo**

* Bài báo giới thiệu thuật toán NRAGLS (News Recommendation Algorithm Based on Gated Linear Attention and Simplified Gated Linear Units) nhằm nâng cao độ chính xác và cá nhân hóa trong việc đề xuất tin tức. * Mô hình cốt lõi sử dụng hai kỹ thuật tiên tiến là Gated Linear Attention (GLA) và Simplified Gated Linear Units (SGLU), kết hợp với chuẩn hóa tensor để xử lý tính chất đa diện của nội dung tin tức và tương tác của người dùng.


* NRAGLS được thiết kế để giải quyết các thách thức phổ biến trong lĩnh vực đề xuất tin tức, bao gồm sự thưa thớt của dữ liệu, sự tiến hóa động của nội dung tin tức và việc thiếu phản hồi rõ ràng từ người dùng.


* Thuật toán này kết hợp nhiều đặc trưng của tin tức như tiêu đề, danh mục, bản tóm tắt và chủ đề để học một vector biểu diễn thống nhất cho tin tức.


* Mô hình cũng tập trung nắm bắt các đặc trưng sở thích theo thời gian của người dùng thông qua mạng GLA và SGLU, tích hợp cả sở thích dài hạn và ngắn hạn để thiết lập vector biểu diễn người dùng.


* Thông qua việc xác thực thực nghiệm nghiêm ngặt trên tập dữ liệu MIND và biến thể MIND-small, hiệu suất của mô hình NRAGLS đã được so sánh với bốn hệ thống đề xuất hiện có và thể hiện hiệu suất vượt trội.


* Mô hình này cũng cho thấy sự nhất quán đáng kể về hiệu suất ngay cả khi dữ liệu bị thiếu hụt ở các mức độ khác nhau và duy trì hoặc cải thiện độ chính xác theo thời gian.



**Các số liệu đánh giá (Metrics)**

Bài báo sử dụng ba số liệu chính để đánh giá khả năng dự đoán mức độ tương tác của người dùng và xếp hạng bài viết:

* 
**AUC (Area Under Curve)**: Số liệu này cung cấp cái nhìn sâu sắc về khả năng của mô hình trong việc phân biệt giữa các trường hợp tích cực (được nhấp) và tiêu cực (không được nhấp).


* 
**MRR (Mean Reciprocal Rank)**: Được sử dụng để đo lường hiệu quả của mô hình trong việc xếp hạng bài báo mong muốn (được nhấp) ở đầu danh sách đề xuất.


* 
**nDCG**: Xem xét tính chính xác và mức độ liên quan của các mục tin tức được đề xuất, áp dụng mức giảm trừ theo logarit dựa trên vị trí của chúng trong danh sách đề xuất.



Kết quả hiệu suất cao nhất mà mô hình NRAGLS đạt được trong các thử nghiệm:

* Tập dữ liệu MIND: AUC đạt 0.7212, MRR đạt 0.3612 và nDCG đạt 0.4245.


* Tập dữ liệu MIND-small: AUC đạt 0.7212, MRR đạt 0.3612 và nDCG đạt 0.4245.