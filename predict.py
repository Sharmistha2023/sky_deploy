import requests
import click

@click.command()
@click.option('--url', '-url', help='url of the deployment')
@click.option('--token', '-token', help='serving token')
@click.option('--imagefile', '-image', help="absolute path of the image")

def predict(url,token,imagefile):
    updated_url = f"{url}/predict"
    with open(imagefile, "rb") as f:
        img_bytes = f.read()

    headers = {
        "Content-Type": "application/octet-stream",
        "Authorization": f"Bearer {token}"
    }
    response = requests.post(updated_url, data=img_bytes, headers=headers, verify=False)
    result = response.json()
    result = result["class_index"]
    if result == 8:
        print("predicted output: bag")
    elif result == 1:
        print("predicted output: Trouser")
    elif result == 2:
        print("predicted output: pullover")
    elif result == 0:
        print("predicted output: T-shirt/Top")
    elif result == 3:
        print("predicted output: Dress")
    elif result == 4:
        print("predicted output: Coat")
    elif result == 5:
        print("predicted output: sandal")
    elif result == 6:
        print("predicted output: shirt")
    elif result == 7:
        print("predicted output: Sneaker")
    elif result == 9:
        print("predicted output: Ankel-boat")
    else:
        print("Not found")

if __name__ == "__main__":
    predict()
