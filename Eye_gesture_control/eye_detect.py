import pyttsx3
engine = pyttsx3.init()


class Myclass:
    def __int__(self):
        pass

    def Blink(self):
        if self == 1:
            engine.say("Light on")
            engine.runAndWait()
            print("Light on")


        if self == 2:
            engine.say("Light off")
            engine.runAndWait()
            print("Light off")

        if self == 3:
            engine.say("Fan on")
            engine.runAndWait()
            print("Fan on")

        if self == 4:
            engine.say("Fan off")
            engine.runAndWait()
            print("Fan off")

        if self == 5:
            engine.say("bed lamp on")
            engine.runAndWait()
            print("Bed lamp on")

        if self == 6:
            engine.say("Bed lamp off")
            engine.runAndWait()
            print("Bed lamp off")
