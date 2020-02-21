import csv,os,cv2
def convert_img_to_csv(img_dir):
    #设置需要保存的csv路径
    with open("newtest.csv","w",newline="") as f:
        #设置csv文件的列名
        column_name = ["label"]
        column_name.extend(["pixel%d"%i for i in range(28*28)])
        #将列名写入到csv文件中
        writer = csv.writer(f)
        writer.writerow(column_name)
        #该目录下有9个目录,目录名从0-9

        #获取目录的路径
        img_temp_dir = os.path.join(img_dir)
        #获取该目录下所有的文件
        img_list = os.listdir(img_temp_dir)
        #遍历所有的文件名称
        for img_name in img_list:
            #判断文件是否为目录,如果为目录则不处理
            if not os.path.isdir(img_name):
                #获取图片的路径
                img_path = os.path.join(img_temp_dir,img_name)
                #因为图片是黑白的，所以以灰色读取图片
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                #图片标签
                row_data = [0]
                #获取图片的像素
                row_data.extend(img.flatten())
                #将图片数据写入到csv文件中
                writer.writerow(row_data)


if __name__ == "__main__":
    #将该目录下的图片保存为csv文件
    convert_img_to_csv(r"F:/DockerProjet/DI5-S10-ProjetLibre-IA/AI_Test/Perceptron/circles")