import json
import cv2

# Automatic FPS Detection
try:
    # Detect video fps by the input video
    video = cv2.VideoCapture('video2_evaluation.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    print(fps)
    video.release()
except:
    fps = 25
VIDEO_FPS = fps  # Your video fps default


# Run for SN ball , SN(TWO MODELS = TWO RUNS):
for i in range(2):
    inp = 'inference_output/results_soccernet.json' if i==1 else 'inference_output/results_snball.json'
    output = 'inference_output/results_sn.txt' if i==1 else 'inference_output/results_snball.txt' 
    with open(inp, 'r') as f:
        data = json.load(f)

    print("Time\t\tLabel\t\tConfidence")
    print("-" * 50)
 
    for pred in data['predictions']:
        frame = pred['frame']
        time_sec = (frame / VIDEO_FPS) + 0.5  # NO stride multiplication!
    
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
    
        # print(f"{minutes:02d}:{seconds:05.2f}\t{pred['label']:<20}\t{pred['confidence']:.3f}")
        # Write it into a file line after line
        with open(output, 'a') as f:
            f.write(f"{minutes:02d}:{seconds:05.2f}\t{pred['label']:<20}\t{pred['confidence']:.3f}\n")