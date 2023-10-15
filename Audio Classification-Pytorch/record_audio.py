import time

from macls.utils.record import RecordAudio

s = input('Please enter how many seconds you plan to record:')
record_seconds = int(s)
save_path = "dataset/save_audio/%s.wav" % str(int(time.time()*1000))

record_audio = RecordAudio()
input(f"Press the Enter key to switch on the recording, recording{record_seconds}seconds：")
record_audio.record(record_seconds=record_seconds,
                    save_path=save_path)

print('The file is saved in:%s' % save_path)
