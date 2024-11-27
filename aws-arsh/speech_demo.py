import pyttsx3
engine = pyttsx3.init()  # object creation

""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
print(rate)  # printing current voice rate
engine.setProperty('rate', 200)     # setting up new voice rate


"""VOLUME"""
volume = engine.getProperty(
    'volume')  # getting to know current volume level (min=0 and max=1)
print(volume)  # printing current volume level
engine.setProperty('volume', 1.0)    # setting up volume level  between 0 and 1

"""VOICE"""
voices = engine.getProperty('voices')  # getting details of current voice
# engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
# changing index, changes voices. 1 for female
engine.setProperty('voice', "com.apple.speech.synthesis.voice.samantha")
# for voice in voices:
#     print("Voice:")
#     print("ID: %s" % voice.id)
#     print("Name: %s" % voice.name)
#     print("Age: %s" % voice.age)
#     print("Gender: %s" % voice.gender)
#     print("Languages Known: %s" % voice.languages)
engine.say("Hello Arshdeep!")
engine.say('We see that there are two voices by default, a male and a female one. The synthesizer sets the male voice as default if not specified otherwise. ' + str(rate))
engine.runAndWait()
engine.stop()

"""Saving Voice to a file"""
# On linux make sure that 'espeak' and 'ffmpeg' are installed
engine.save_to_file('Hello World', 'test.mp3')
engine.runAndWait()
