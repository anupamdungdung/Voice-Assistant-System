from tkinter import *
from brain import *

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"


class ChatApplication:

    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Voice Assistant")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_COLOR)

        # head label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Welcome! How can I help you?", font=FONT_BOLD, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=BG_GRAY)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # # text widget
        # self.text_widget = Text(self.window, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
        #                         font=FONT, padx=5, pady=5)
        # self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        # self.text_widget.configure(cursor="arrow", state=DISABLED)

        # bottom label
        bottom_label = Label(self.window, bg=BG_GRAY, height=80)
        bottom_label.place(relwidth=1, rely=0.825)

        # button widget
        b2 = Button(bottom_label, text="Press to Start")
        b2.place(relx=0.5, rely=0.5,relwidth=1, anchor=CENTER)


if __name__ == "__main__":
    app = ChatApplication()
    app.run()

# r = Tk()  # Root of the app
# r.title('Voice Assistant')
# w = Canvas(r, width=400, height=400)
# w.pack()
# button = Button(r, text='Press to Start', width=25, command=brain)
# button.pack()
# button = Button(r, text='Stop', width=25, command=exitTheApp)
# button.pack()
# r.mainloop()
