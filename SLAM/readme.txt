test1 ԭͼ
test2 ԭͼ(����)
test3 JPEG(�ɣ�����)
test4 JPEG(�£�����)
test5 GRACE(����)
test6 PNG
test7 PNG(����)
test8 BMP
test9 BMP(����)
test10 WebP
test11 WebP(����)
test12 H.264

test13 H.264(����)

ffmpeg -y -f image2 -i test1.jpeg -vcodec libx264 -r 1 -t 1 test12.mp4
ffmpeg -i test12.mp4 -r 1 -f image2 test12.jpeg