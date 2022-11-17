import os
import cv2

def imgs2video(img_root, video_path, fps, size, s, e): # 仅将s到e的图转为视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter(video_path, fourcc, fps, size)
    for i in range(s, e):  # 有多少张图片，从编号s到编号e
        print(i)
        img = cv2.imread(img_root + "%05d"%i + '.png')
        # cropped_img = img[0:960, 0:720] # 裁剪图片
        videoWriter.write(img)
    videoWriter.release()

if __name__ == "__main__":
    video_path = 'D:/videos/'  # 视频所在的路径，视频放在video文件夹下
    f_save_path = 'D:/02_Project/sky111/'
    img_root = f_save_path + "exp11/" # 图片存储的路径 /imgs/test/
    video_save_path = f_save_path + "test.avi" # 视频保存路径
    img_len = len(os.listdir(img_root)) # 图片数量
    imgs2video(img_root, video_save_path, 25, (800, 640), 0, img_len)
    print("imgs转video成功！")``