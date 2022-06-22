import os
import shutil
import time
import torch
import re
import ffmpeg
from pathlib import Path
from razdel import sentenize
from wand.image import Image
from wand.color import Color

language = "ru"
model_id = "v3_1_ru"
speaker = "xenia"
device = torch.device("cpu")
torch.set_num_threads(4)
sample_rate = 48000
base_filename = "ru_" + time.strftime("%Y%m%d_%H%M%S")

# Регулярка для поиска строки с названием файла
pattern_filename = re.compile("={5} (.+?) ={5}")
DIR = Path(__file__).resolve().parent / "tmp" / base_filename
DIR.mkdir(exist_ok=True)

with open("app/ru_example.txt", "r", encoding="windows-1251") as f:
    file_arr = []
    file_count = 0
    filename = ""
    file_obj = None
    is_file_header = True

    for line in f:
        # Проверка наличия заголовка в файле
        match = pattern_filename.search(line)
        if match:
            if file_obj:
                file_obj.close()

            file_count+=1
            print("+ Создание нового файла: " + str(match.group(1)) + str(file_count))
            file_arr.append(str(match.group(1)) + str(file_count))
            filename = DIR / (str(match.group(1)) + str(file_count) + ".txt")
            file_obj = open(filename, "w", encoding="windows-1251")
            file_obj.write("<break time='5s'/>")
            is_file_header = True
            continue

        if not filename:
            continue

        if is_file_header:
            file_obj.write(line)
            is_file_header = False

        else:
            file_obj.write(line)

    if file_obj:
        file_obj.close()

print(file_arr)
print("# Количество страниц для озвучки: " + str(file_count))

model, example_text = torch.hub.load(repo_or_dir="snakers4/silero-models",
                                     model="silero_tts",
                                     language=language,
                                     speaker=model_id)
model.to(device)

def audioparse(text):
    result = []
    lines = text.splitlines()
    for i in lines:
        if 0 < len(i) <= 1000:
            print("Длина строки: " + str(len(i)))
            temp_text = "<speak><p>" + str(i) + "</p></speak>"
            result.append(model.apply_tts(ssml_text=temp_text,
                                speaker=speaker,
                                sample_rate=sample_rate))
        elif len(i) > 1000:
            print("Длина строки: " + str(len(i)))
            for sentence in list(sentenize(i)):
                result.append(model.apply_tts(ssml_text="<speak>" + str(sentence.text) + "</speak>",
                                              speaker=speaker,
                                              sample_rate=sample_rate))
        else:
            print("Ошибка длины строки!")
    return result

for pagenum in file_arr:
    audio = torch.empty
    file = DIR / (pagenum + ".txt")
    with open(file, "r", encoding="windows-1251") as f:
        page_text = f.read()
    audio = torch.cat(audioparse(page_text))
    model.write_wave(path="app/tmp/" + base_filename + "/" + pagenum + ".wav",
                    audio=(audio * 32767).numpy().astype("int16"),
                    sample_rate=sample_rate)
    print("+ Записано новое аудио для страницы: " + pagenum)

def convert_pdf(filename, output_path="", resolution=300):
    all_pages = Image(filename=filename, resolution=resolution)
    for i, page in enumerate(all_pages.sequence):
        with Image(page) as img:
            img.format = "png"
            img.transform(resize='2560x')
            img.background_color = Color("white")
            # img.alpha_channel = "remove"

            image_filename = os.path.splitext(os.path.basename(filename))[0]
            image_filename = "PAGE{}.png".format(i+1)
            image_filename = os.path.join(output_path, image_filename)

            img.save(filename=image_filename)

convert_pdf(filename="app/ru_example.pdf", output_path="app/tmp/" + base_filename)

def page_to_mp4(filepath, filename):

    image_file = ffmpeg.input(filepath + "/" + filename + ".png", r=1)
    audio_file = ffmpeg.input(filepath + "/" + filename + ".wav")
    (
        ffmpeg
        .output(image_file, audio_file, filepath + "/" + filename + ".mp4", vcodec="h264_amf", acodec="libmp3lame", audio_bitrate="320k")
        .run()
    )

for pagenum in file_arr:
    page_to_mp4(filepath="app/tmp/" + base_filename, filename=pagenum)

def united_mp4(filepath, out_filename, page_arr):
    file_obj = open(filepath + "/pages.txt", "w", encoding="windows-1251")
    for page_name in page_arr:
        file_obj.write("file '" + page_name+".mp4'\n")
    file_obj.close()
    (
        ffmpeg
        .input(filepath + "/pages.txt", f="concat")
        .output("app/ready/" + out_filename + ".mp4", vcodec="copy", acodec="copy")
        .run()
    )

united_mp4(filepath="app/tmp/" + base_filename, out_filename=base_filename, page_arr=file_arr)

def clean_tmp(filepath):
    try:
        shutil.rmtree(filepath)
    except OSError as e:
        print("Ошибка очистки временной папки: %s - %s." % (e.filename, e.strerror))

clean_tmp(filepath="app/tmp/" + base_filename)