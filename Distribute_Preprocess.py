scene = 0
num_scenes = 74
num_scenes_to_process = round(num_scenes/num_threads)
threads = []
for i in range(num_threads):
    if (i == num_threads - 1):
        num_scenes_to_process = num_scenes - scene
    thread = Thread(target=preprocess, args=(scene, num_scenes_to_process))
    thread.start()
    threads.append(thread)
    scene += num_scenes_to_process

for thread in threads:
    thread.join()