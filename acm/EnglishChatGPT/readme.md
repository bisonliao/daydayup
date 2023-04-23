一个简陋的用chatGPT的能力进行英语口语练习的app和它的后台。

1. app录音，传输到后台
2. 调用ffmpeg转一下封装
3. 调用openai的whisper接口识别出文字
4. 调用openai的chat接口聊天，得到应答
5. 调用腾讯云TTS服务将文字转语音
6. 将语音发回app，播放出来
7. app与后端服务使用websocket通信