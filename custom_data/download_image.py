from simple_image_download import simple_image_download as simp  # Should use simple_image_donwload version 0.4

response = simp.simple_image_download
keywords = ["slam eagle", "제공호"]

for kw in keywords:
    response().download(kw, 200)