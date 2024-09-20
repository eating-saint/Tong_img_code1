import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time

start_time = time.time()

def inspect(sequence,p):
    best_sequence = []
    current_sequence = []
    for i in range(1, len(sequence)):
        if sequence[i] - sequence[i-1] < len(sequence) // p:
            current_sequence.append(sequence[i])
        else:
            if current_sequence: # �����ǰ���в�Ϊ��
                if len(best_sequence) == 0 or len(current_sequence) > len(best_sequence):
                    best_sequence = current_sequence
            current_sequence = [sequence[i]] # ���¿�ʼ�µ�����
    # �������һ������
    if current_sequence: # �����ǰ���в�Ϊ��
        if len(best_sequence) == 0 or len(current_sequence) > len(best_sequence):
            best_sequence = current_sequence
    return best_sequence

def shadow_cheak_L(L,total):
    if L > 5:
        compare = int(np.sum(total[int(L):(int(L)+5)]))
        compare_s = int(np.sum(total[(int(L)-5):int(L)]))
        if (compare - compare_s)//5 < 10:
            return [0]
        else:
            return L
    else:
        return L

def shadow_cheak_R(R,total,tail):
    if  tail - R > 5:
        compare = int(np.sum(total[int(R):int(R)+5]))
        compare_s = int(np.sum(total[int(R)-5:int(R)]))
        if (compare_s - compare)//5 < 10: 
            return [tail]
        else:
            return R
    else:
        return R

def center_choose(best_sequence,column_values,n):
    if best_sequence[0] < 3:
        x = best_sequence[0]
    else:
        x = shadow_cheak_L(best_sequence[0],column_values)
    if n - best_sequence[-1] < 3:
        y = best_sequence[-1]
    else:
        y = shadow_cheak_R(best_sequence[-1],column_values,n)
    return x,y

def read_picture(file_path):
    gray = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)  # ת��Ϊ�Ҷ�ͼ��
    image = gray
    return image,gray

def bit_leave_not(image):
    nrow = image.shape[0]
    ncol = image.shape[1]
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh, nrow, ncol

def line_four_cheak_L(image, thresh, ncol):
    threshold = 40
    column_values = thresh[:, ncol // 3]
    column_values_1 = thresh[:, ncol * 2 // 3]
    fg = np.argwhere(column_values > threshold)
    fg_1 = np.argwhere(column_values_1 > threshold)
    return fg, fg_1,threshold

def ox_L(fg,fg_1,image,ncol):
    if len(fg) > len(fg_1):
        best_sequence = inspect(fg, 20)
        column_values1 = image[:, ncol // 3]
    else:
        best_sequence = inspect(fg_1, 20)
        column_values1 = image[:, ncol * 2 // 3]
        fg = fg_1
    return best_sequence, column_values1, fg 

def kL(nrow, best_sequence, column_values1, fg):
    if len(best_sequence) > len(fg) * 4 // 5:
        xL, yL = center_choose(best_sequence, column_values1, nrow)
    else:
        xL, yL = center_choose(fg, column_values1, nrow)
    return xL,yL

def line_four_cheak_R(image, thresh, nrow, threshold):
    column_values = thresh[nrow // 3, :]
    column_values_1 = thresh[nrow * 2 // 3, :]
    fg = np.argwhere(column_values > threshold)
    fg_1 = np.argwhere(column_values_1 > threshold)
    return fg,fg_1

def ox_R(fg,fg_1,image,nrow):
    if len(fg) > len(fg_1):
        best_sequence = inspect(fg, 20)
        column_values1 = image[nrow // 3, :]
    else:
        best_sequence = inspect(fg_1, 20)
        column_values1 = image[nrow * 2 // 3, :]
        fg = fg_1
    return best_sequence, column_values1, fg

def kR(ncol, best_sequence, fg, column_values1):
    if len(best_sequence) > len(fg) * 4 // 5:
        xR, yR = center_choose(best_sequence, column_values1, ncol)
    else:
        xR, yR = center_choose(fg, column_values1, ncol)
    return xR, yR

def save(file_path, xL, yL, xR, yR, image):
    cv2.imencode('.bmp', image[int(xL[0]):int(yL[0]), int(xR[0]):int(yR[0])])[1].tofile(file_path)
    return True

def process_image(file_path):
    try:
        with ThreadPoolExecutor() as executor:
            image,gray = executor.submit(read_picture, file_path).result()
            thresh, nrow, ncol = executor.submit(bit_leave_not, image).result()
            fg, fg_1, threshold = executor.submit(line_four_cheak_L, image, thresh, ncol).result()        
            best_sequence, column_values1, fg = executor.submit(ox_L, fg, fg_1, image, ncol).result()
            xL,yL = executor.submit(kL, nrow, best_sequence, column_values1, fg).result()
            fg,fg_1 = executor.submit(line_four_cheak_R,image, thresh, nrow, threshold).result()
            best_sequence, column_values1, fg = executor.submit(ox_R, fg, fg_1, image, nrow).result()
            xR,yR = executor.submit(kR, ncol, best_sequence, fg, column_values1).result()
            executor.submit(save, file_path, xL, yL, xR, yR, gray).result()
        return True
    except :
        print("ERROR������")
        print(file_path)

if __name__ == '__main__':
    # ����Ҫ������ļ���·��
    folder_path = r'./21�Ŀ�1'
    cnt = 0
    # ��ȡ�ļ����е������ļ���
    file_names = os.listdir(folder_path)
    # �����ļ��б�����ÿ���ļ�
    for file_name1 in file_names: #��һ���ļ���
        # ��ȡ�ļ�·��
        file_path1 = folder_path + '/' + file_name1
        file_names1 = os.listdir(file_path1)
        for file_name2 in file_names1:  #�ڶ����ļ���
        # ��ȡ�ļ�·��
            file_path2 = file_path1 + '/' + file_name2
            file_names2 = os.listdir(file_path2)
            for file_name3 in file_names2:  #�������ļ���
        # ��ȡ�ļ�·��
                file_path3 = file_path2 + '/' + file_name3
                pts = file_path3
                if process_image(pts):
                    cnt = cnt + 1
                    print(cnt)
                else:
                    print("��ͼƬ�����������ϴ�ʵ����Ƭ")
cv2.waitKey(0)
cv2.destroyAllWindows()

end_time = time.time()
# ����������и�ͼ�����ʱ��
total_time = end_time - start_time
print(f"Total time: {total_time} seconds")

