from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.relativelayout import RelativeLayout 
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Line, Rectangle
from image_to_text import get_text, train_model
import cv2
from string import ascii_uppercase as UC



class DrawWidget(RelativeLayout):

    def __init__(self, **kwargs):

        super(DrawWidget, self).__init__(**kwargs)

        with self.canvas:
           Color(*(1,1,1),mode="rgb")
           
           self.rect = Rectangle(size = self.size, pos = self.pos)
        self.bind(size=self.update_rect)

    def on_touch_down(self, touch):
        color = (0,0,0)
        with self.canvas:
            Color(*color,mode="rgb")
            width = 2.5
            x,y = self.to_local(x=touch.x, y=touch.y)
            touch.ud["line"] = Line(points=(x, y),width=width)
    def on_touch_move(self,touch):
        x,y = self.to_local(x=touch.x, y=touch.y)
        touch.ud['line'].points += [x, y]

    def update_rect(self, instance, value):
        self.rect.size = self.size
        self.rect.pos = self.pos

class DrawApp(App):
    
    def build(self):
        self.title = 'Convert To Text'

        parent = RelativeLayout()
        

        self.draw = DrawWidget(size_hint=(0.5,0.8),pos_hint={'x':0,'y':0.2})

        clear_btn = Button(size_hint=(0.5,0.1),text="Clear",pos_hint={'x':0,'y':0.1})
        clear_btn.bind(on_release=self.clear_canvas)

        convert_btn = Button(size_hint=(0.5,0.1),text="Convert to text",pos_hint={'x':0.5,'y':0.1})
        convert_btn.bind(on_release=self.convert)

        self.label = Label(size_hint=(0.5,0.9),pos_hint={'x':0.5,'y':0.2})

        label1 = Label(size_hint=(0.3,0.1),pos_hint={"x":0,"y":0},text="Wrong conversion? Type in correct capital letters comma separated and train")
        label1.bind(width=lambda *x: label1.setter('text_size')(label1, (label1.width, None)), texture_size=lambda *x: label1.setter('height')(label1, label1.texture_size[1]))

        self.inp_txt = TextInput(size_hint=(0.4,0.1),pos_hint={"x":0.3,"y":0})
        self.train_btn = Button(size_hint=(0.3,0.1),pos_hint={"x":0.7,"y":0},text="Train", disabled=True)
        self.train_btn.bind(on_release = self.train)

        parent.add_widget(self.draw)
        parent.add_widget(self.label)
        parent.add_widget(clear_btn)
        parent.add_widget(convert_btn)
        parent.add_widget(label1)
        parent.add_widget(self.inp_txt)
        parent.add_widget(self.train_btn)


        return parent

    def clear_canvas(self, obj):
        self.draw.canvas.clear()
        with self.draw.canvas:
           Color(*(1,1,1),mode="rgb")
           self.draw.rect = Rectangle(size = self.draw.size, pos = (0,0))
        self.draw.bind(size=self.draw.update_rect)
        self.train_btn.disabled = True

    def convert(self, obj):
        self.train_btn.disabled = False
        self.draw.export_to_png("draw.png")
        img = cv2.imread("draw.png")
        self.lets, self.imgs = get_text(img)
        txt = " ".join(self.lets)
        self.label.text = txt

    def train(self, obj):
        let = self.inp_txt.text
        let = let.replace(" ","").split(",")
        lbls = []
        chars = list(UC)
        for l in let:
            lbls.append(chars.index(l))
        if len(lbls) == len(self.imgs):
            train_model(self.imgs, lbls)

if __name__ == "__main__":
    DrawApp().run()    

