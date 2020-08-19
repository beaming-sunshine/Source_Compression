test1 原图
test2 原图(处理)
test3 JPEG(旧，处理)
test4 JPEG(新，处理)
test5 GRACE(处理)
test6 PNG
test7 PNG(处理)
test8 BMP
test9 BMP(处理)
test10 WebP
test11 WebP(处理)
test12 H.264

test13 H.264(处理)

ffmpeg -y -f image2 -i test1.jpeg -vcodec libx264 -r 1 -t 1 test12.mp4
ffmpeg -i test12.mp4 -r 1 -f image2 test12.jpeg