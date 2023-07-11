import tkinter as tk
from keras.models import load_model
import cv2
import numpy as np
from tkinter import *

# Load the trained model
model = load_model("mnist_cnn.h5")

def predict_digit(img):
    img = img.reshape(1, 28, 28, 1)
    img = img.astype("float32") / 255
    return np.argmax(model.predict(img))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognition")
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white")
        self.canvas.pack()
        self.bind("<B1-Motion>", self.draw)
        self.btn_recognize = tk.Button(self, text="Recognize", command=self.recognize)
        self.btn_recognize.pack()
        self.btn_clear = tk.Button(self, text="Clear", command=self.clear)
        self.btn_clear.pack()
        self.image = np.zeros((300, 300), dtype=np.uint8)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="black", width=0)
        self.image[y - 5:y + 5, x - 5:x + 5] = 255

    def recognize(self):
        img = cv2.resize(self.image, (28, 28), interpolation=cv2.INTER_AREA)
        digit = predict_digit(img)
        print("Recognized digit:", digit)

    def clear(self):
        self.canvas.delete("all")
        self.image.fill(0)

if __name__ == "__main__":
    app = App()
    app.mainloop()
