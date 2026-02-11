import os
import json
import subprocess
import sys

dataset_path = r"C:\Users\ROG\.cache\huggingface\lerobot\joyandai\lerobot_v3"
video_files = [
    os.path.join(dataset_path, "videos", "observation.images.fixed", "chunk-000", "file-000.mp4"),
    os.path.join(dataset_path, "videos", "observation.images.handeye", "chunk-000", "file-000.mp4")
]
info_json_path = os.path.join(dataset_path, "meta", "info.json")

new_width = 640
new_height = 360

print(f"Starting resize process for {len(video_files)} files...")

# 1. Resize videos
for f in video_files:
    if not os.path.exists(f):
        print(f"Skipping missing file: {f}")
        continue
    
    tmp_out = f.replace(".mp4", "_tmp.mp4")
    print(f"Resizing {f} to {new_width}x{new_height}...")
    
    cmd = [
        "ffmpeg", "-y", "-i", f,
        "-vf", f"scale={new_width}:{new_height}",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-stats",
        tmp_out
    ]
    
    try:
        # Run and let it print to terminal
        subprocess.run(cmd, check=True)
            
        if os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
            if os.path.exists(f):
                os.remove(f)
            os.rename(tmp_out, f)
            print(f"Successfully finished: {f}")
        else:
            print(f"Error: Temporary file {tmp_out} is empty or missing.")
    except Exception as e:
        print(f"Failed to process {f}: {e}")
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except:
                pass

# 2. Update info.json
if os.path.exists(info_json_path):
    print(f"Updating {info_json_path}...")
    try:
        with open(info_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        keys = ["observation.images.fixed", "observation.images.handeye"]
        for k in keys:
            if k in data.get("features", {}):
                data["features"][k]["shape"] = [new_height, new_width, 3]
                if "info" in data["features"][k]:
                    data["features"][k]["info"]["video.height"] = new_height
                    data["features"][k]["info"]["video.width"] = new_width
                print(f"Updated metadata for {k}")
                
        with open(info_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("info.json metadata updated successfully.")
    except Exception as e:
        print(f"Failed to update info.json: {e}")
else:
    print(f"Warning: {info_json_path} not found.")

print("All done!")
