import streamlit as st
import base64

def get_base64_image(image_path):
    with open(image_path, 'rb') as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')
    return b64_string

# Path to your local image file
local_image_path = 'user.png'

# Get the base64 string of the local image
user_image_base64 = get_base64_image(local_image_path)

# HTML template with the embedded base64 image


css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/png;base64,{user_image_base64}">
    </div>    
    <div class="message">{{{{MSG}}}}</div>
</div>
'''


