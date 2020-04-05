import csv,os,cv2
def convert_img_to_csv(img_dir):
    #设置需要保存的csv路径
    with open("imagePixel.csv","w",newline="") as f:
        #设置csv文件的列名
        column_name = ["label"]
        column_name.extend(["pixel%d"%i for i in range(28*28)])
        #将列名写入到csv文件中
        writer = csv.writer(f)
        writer.writerow(column_name)

        img_dir_circle = img_dir + 'circles'
        img_dir_square = img_dir + 'squares'

        #获取目录的路径
        img_temp_dir_circle = os.path.join(img_dir_circle)
        #获取该目录下所有的文件
        img_list_circle = os.listdir(img_temp_dir_circle)
        # #遍历所有的文件名称
        # for img_name_circle in img_list_circle:
        #     #判断文件是否为目录,如果为目录则不处理
        #     if not os.path.isdir(img_name_circle):
        #         #获取图片的路径
        #         img_path = os.path.join(img_temp_dir_circle,img_name_circle)
        #         #因为图片是黑白的，所以以灰色读取图片
        #         img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        #         #图片标签
        #         row_data = [0]
        #         #获取图片的像素
        #         row_data.extend(img.flatten())
        #         #将图片数据写入到csv文件中
        #         writer.writerow(row_data)


        img_temp_dir_square = os.path.join(img_dir_square)
        #获取该目录下所有的文件
        img_list_square = os.listdir(img_temp_dir_square)
        # #遍历所有的文件名称
        # for img_name_square in img_list_square:
        #     #判断文件是否为目录,如果为目录则不处理
        #     if not os.path.isdir(img_name_square):
        #         #获取图片的路径
        #         img_path = os.path.join(img_temp_dir_square,img_name_square)
        #         #因为图片是黑白的，所以以灰色读取图片
        #         img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        #         #图片标签
        #         row_data = [1]
        #         #获取图片的像素
        #         row_data.extend(img.flatten())
        #         #将图片数据写入到csv文件中
        #         writer.writerow(row_data)

        for img_name_circle, img_name_square in zip(img_list_circle, img_list_square):
            # 判断文件是否为目录,如果为目录则不处理
            if not os.path.isdir(img_name_circle):
                #获取图片的路径
                img_path = os.path.join(img_temp_dir_circle,img_name_circle)
                #因为图片是黑白的，所以以灰色读取图片
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                #图片标签
                row_data = [0]
                #获取图片的像素
                row_data.extend(img.flatten())
                #将图片数据写入到csv文件中
                writer.writerow(row_data)
            #判断文件是否为目录,如果为目录则不处理
            if not os.path.isdir(img_name_square):
                #获取图片的路径
                img_path = os.path.join(img_temp_dir_square,img_name_square)
                #因为图片是黑白的，所以以灰色读取图片
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                #图片标签
                row_data = [1]
                #获取图片的像素
                row_data.extend(img.flatten())
                #将图片数据写入到csv文件中
                writer.writerow(row_data)

if __name__ == "__main__":
    #将该目录下的图片保存为csv文件
    #convert_img_to_csv(r"F:/DockerProjet/DI5-S10-ProjetLibre-IA/AI_Test/Perceptron/circles")
    convert_img_to_csv("../Dataset/basicshapes/")