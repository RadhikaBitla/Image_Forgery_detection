from tkinter import *;
#from PIL import ImageTk, Image
import image;
import numpy as np;
import cv2;
image_path="";

def task(event):
        button1.config(bg='Aqua')
        #if(textbox1.get()=='')
        image_path = textbox1.get()+".png"
        quantization = 16
        tsimilarity = 5  # euclid distance similarity threshhold
        tdistance = 20  # euclid distance between pixels threshold
        vector_limit = 20  # shift vector elimination limit
        block_counter = 0
        block_size = 8

        imageName = image_path;
        image = cv2.imread(imageName)
        mask = cv2.imread('forged2_mask.png')
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        temp = []
        arr = np.array(gray)
        mask = np.array(mask_gray)
        prediction_mask = np.zeros((arr.shape[0], arr.shape[1]))
        column = arr.shape[1] - block_size
        row = arr.shape[0] - block_size
        dcts = np.empty((((column + 1) * (row + 1)), quantization + 2))

        # ----------------------------------------------------------------------------------------

        print("scanning & dct starting...")

        for i in range(0, row):
            for j in range(0, column):

                blocks = arr[i:i + block_size, j:j + block_size]
                imf = np.float32(blocks) / 255.0  # float conversion/scale
                dst = cv2.dct(imf)  # the dct
                blocks = np.uint8(np.float32(dst) * 255.0)  # convert back
                # zigzag scan
                solution = [[] for k in range(block_size + block_size - 1)]
                for k in range(block_size):
                    for l in range(block_size):
                        sum = k + l
                        if (sum % 2 == 0):
                            # add at beginning
                            solution[sum].insert(0, blocks[k][l])
                        else:
                            # add at end of the list
                            solution[sum].append(blocks[k][l])

                for item in range(0, (block_size * 2 - 1)):
                    temp += solution[item]

                temp = np.asarray(temp, dtype=np.float64)
                temp = np.array(temp[:16])
                temp = np.floor(temp / quantization)
                temp = np.append(temp, [i, j])

                np.copyto(dcts[block_counter], temp)

                block_counter += 1
                temp = []

        print("scanning & dct over!")

        # ----------------------------------------------------------------------------------------

        print("lexicographic ordering starting...")

        dcts = dcts[~np.all(dcts == 0, axis=1)]
        dcts = dcts[np.lexsort(np.rot90(dcts))]

        print("lexicographic ordering over!")

        # ----------------------------------------------------------------------------------------

        print("euclidean operations starting...")

        sim_array = []
        for i in range(0, block_counter):
            if i <= block_counter - 10:
                for j in range(i + 1, i + 10):
                    pixelsim = np.linalg.norm(dcts[i][:16] - dcts[j][:16])
                    pointdis = np.linalg.norm(dcts[i][-2:] - dcts[j][-2:])
                    if pixelsim <= tsimilarity and pointdis >= tdistance:
                        sim_array.append([dcts[i][16], dcts[i][17], dcts[j][16], dcts[j][17], dcts[i][16] - dcts[j][16],
                                          dcts[i][17] - dcts[j][17]])
            else:
                for j in range(i + 1, block_counter):
                    pixelsim = np.linalg.norm(dcts[i][:16] - dcts[j][:16])
                    pointdis = np.linalg.norm(dcts[i][-2:] - dcts[j][-2:])
                    if pixelsim <= tsimilarity and pointdis >= tdistance:
                        sim_array.append([dcts[i][16], dcts[i][17], dcts[j][16], dcts[j][17], dcts[i][16] - dcts[j][16],
                                          dcts[i][17] - dcts[j][17]])

        print("euclidean operations over!")

        # ----------------------------------------------------------------------------------------

        print("elimination starting...")

        sim_array = np.array(sim_array)
        delete_vec = []
        vector_counter = 0
        for i in range(0, sim_array.shape[0]):
            for j in range(1, sim_array.shape[0]):
                if sim_array[i][4] == sim_array[j][4] and sim_array[i][5] == sim_array[j][5]:
                    vector_counter += 1
            if vector_counter < vector_limit:
                delete_vec.append(sim_array[i])
            vector_counter = 0

        delete_vec = np.array(delete_vec)
        delete_vec = delete_vec[~np.all(delete_vec == 0, axis=1)]
        delete_vec = delete_vec[np.lexsort(np.rot90(delete_vec))]

        for item in delete_vec:
            indexes = np.where(sim_array == item)
            unique, counts = np.unique(indexes[0], return_counts=True)
            for i in range(0, unique.shape[0]):
                if counts[i] == 6:
                    sim_array = np.delete(sim_array, unique[i], axis=0)

        print("elimination over!")

        # ----------------------------------------------------------------------------------------

        print("painting starting...")

        for i in range(0, sim_array.shape[0]):
            index1 = int(sim_array[i][0])
            index2 = int(sim_array[i][1])
            index3 = int(sim_array[i][2])
            index4 = int(sim_array[i][3])
            for j in range(0, 7):
                for k in range(0, 7):
                    prediction_mask[index1 + j][index2 + k] = 255
                    prediction_mask[index3 + j][index4 + k] = 255

        print("painting over!")

        # ----------------------------------------------------------------------------------------

        print("accuracy calculating...")

        TP = 0
        FP = 0
        TN = 0
        FN = 0

        for i in range(0, prediction_mask.shape[0]):
            for j in range(0, prediction_mask.shape[1]):
                if prediction_mask[i][j] == mask[i][j]:
                    if prediction_mask[i][j] == 255:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if prediction_mask[i][j] == 255:
                        FP += 1
                    else:
                        FN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = 2 * precision * recall / (precision + recall)

        print('Accuracy:', accuracy)

        print("accuracy calculated!")
        if (accuracy < 70):
            print("Image is forged as accuray is low")
        else:
            print("Image is not forged")

        # ----------------------------------------------------------------------------------------

        cv2.imshow("Result:", prediction_mask)
        # cv2.imshow("Real Mask", mask)
        cv2.imshow('Original Image', image)
        # cv2.imshow('Gray Image', gray)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

window = Tk()
window.title("Image Forgery Detector")
window.geometry('600x400')

bg=PhotoImage(file="C:\\Users\\radhika\\Pictures\\b.png")
myLabel = Label(window, image=bg)
myLabel.place(x=0,y=0,relwidth=1, relheight=1)

label1 = Label(window, text='Image Name', fg='black', font=('Gadugi', 15),bg='white')
label1.place(x=100,y=100)

note1=Label(window,text='NOTE: If result image contains white colour', fg='red',font=('Gadugi',15),bg='Aqua')
note2=Label(window, text='It indicates that, it is a forged image',fg='red',font=('Gadugi',15),bg='Aqua')
note1.place(x=20,y=320)
note2.place(x=80,y=360)
textbox1 = Entry(window, fg='brown', font=('Gadugi', 15))
textbox1.place(x=250, y=100)
button1 = Button(window, text='Submit', fg='black', font=('Gadugi', 15))
button1.place(x=300, y=200)

button1.bind('<Button-1>', task)

button1.mainloop()

label = Label()

window.mainloop()

from pip._internal import main
try:
    import cv2
except Exception as e:
    main(["install", "opencv-python"])
finally:
    pass