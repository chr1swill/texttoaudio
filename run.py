import time
from scipy.io.wavfile import write
import sys
import torch
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Kokoro-82M'))
from kokoro import generate
from models import build_model

if len(sys.argv) != 3:
    argc = len(sys.argv)
    print(f"argc: {argc}")
    print(
        "Ussage: python3 run.py 'input string' [path/to/output/file.wav")
    exit()
else:
    text = sys.argv[1]
    output_wav = sys.argv[2]
    print(f'text={text}')
    print(f'output_pdf="{output_wav}"')

words = text.split()
segments = []

MAX_WORDS = 50

for i in range(0, len(words), MAX_WORDS):
    segment = ' '.join(words[i:i + MAX_WORDS])
    segments.append(segment)
    # print(f"segment: {segment}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

MODEL = build_model('Kokoro-82M/kokoro-v0_19.pth', device)
# you can easily test voices here
# -> https://huggingface.co/spaces/hexgrad/Kokoro-TTS
VOICE_NAME = [
    'af',
    'af_bella', 'af_sarah', 'am_adam', 'am_michael',
    'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis',
    'af_nicole', 'af_sky',
][4]    # 0 (f) and 4 (m) sound the best to me
VOICEPACK = torch.load(
    f'Kokoro-82M/voices/{VOICE_NAME}.pt', weights_only=True).to(device)
print(f'Loaded voice: {VOICE_NAME}')
wav_pieces_dir = "./wav-pieces/"
ffmpeg_inputs_file_path = wav_pieces_dir + "myinput.txt"
try:
    os.stat(ffmpeg_inputs_file_path)
    os.remove(ffmpeg_inputs_file_path)
    print(f"File already exists, removing it: {ffmpeg_inputs_file_path}")
except:
    print(f"Creating file: {ffmpeg_inputs_file_path}")
    pass

comment = "# " + time.strftime("%Y-%m-%d--%H-%M") + \
    " Audio inputs to join together\n"

try:
    os.makedirs(wav_pieces_dir)
except:
    pass

ffmpeg_inputs_file = open(ffmpeg_inputs_file_path, "a")
ffmpeg_inputs_file.write(comment)
print(f"Create file: {ffmpeg_inputs_file_path} with timestamp commnet")

i = 0
print(segments)
for i in range(len(segments)):
    sentence = segments[i]
    print(f"sentence: {sentence}")
    audio, out_ps = generate(MODEL, sentence, VOICEPACK, lang=VOICE_NAME[0], speed=1)
    print(f"audio: {audio}")
    print(f"out_ps: {out_ps}")

    file_path = wav_pieces_dir + str(i) + ".wav"
    write(file_path, rate=24000, data=audio)
    print(f"Wrote audio data to file: {file_path}")

    current_input = "file '" + str(i) + ".wav" + "'\n"
    ffmpeg_inputs_file.write(current_input)
    print(f"Added input file: {file_path} to ffmpeg inputs file")
    i += 1

ffmpeg_inputs_file.close()
ffmpeg_command = "ffmpeg -f concat -safe 0 -i " + ffmpeg_inputs_file_path + \
    " -c copy " + output_wav
print(f"Executing ffmpeg command:\n\t{ffmpeg_command}")
os.system(ffmpeg_command)

# whisper_command = "whisper " + output_path + \
#     "final_output.wav --model tiny --output_dir " + \
#     output_path + "subtitles"
# print(f"Executing whisper command:\n\t{whisper_command}")
# os.system(whisper_command)
#
# tts_filepath = output_path + "final_output.wav"
# subtitle_filepath = output_path + "subtitles/" + "final_output.vtt"
# video_filepath = "$HOME/Videos/stock-clips/purple-fluid-60.mp4"
# music_filepath = "$HOME/Music/royalty-free/caves-of-dawn-10376-reduced-25.mp3"
#
# try:
#     os.makedirs(output_path+"final/")
# except:
#     pass
#
# final_output_filepath = output_path + "final/output.mp4"
#
# tts_duration = f"$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {tts_filepath})"
# subtitle_style = "force_style='Alignment=10,FontSize=18,FontName=Helvetica,Outline=4,OutlineColor=&HFFFFFFFF,PrimaryColor=&HF00000000'"
#
# video_create_command = f"ffmpeg -stream_loop -1 -i {video_filepath} -i {tts_filepath} -i {music_filepath} -lavfi \"[0:v]subtitles={subtitle_filepath}:{subtitle_style}[v];[1:a][2:a]amix=inputs=2:duration=longest[a]\" -map \"[v]\" -map \"[a]\" -t {tts_duration} -c:v libx264 -c:a aac -b:a 192k {final_output_filepath}"
#
# print(f"Executing command to create final video\n\t{video_create_command}")
# os.system(video_create_command)
