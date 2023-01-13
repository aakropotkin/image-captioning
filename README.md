# Image Captioning

captioning is a Python library for dealing with image captioning using BLIP.

## Checkpoints [Required]

Download the fine-tuned checkpoint and copy into 'checkpoints' folder (create if does not exists)

- [BLIP-Large](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth)

## Demo

<img src='./demo.jpg' width=500px>

```txt
./images/image_003.png, a painting of a man in a turban reading a book
./images/image_008.png, a woman in a white and red outfit
./images/image_009.png, a statue of a woman holding a bird
./images/image_015.png, a woman with freckles and flowers in her hair
./images/image_021.png, a painting of a woman's face with fire coming out of her hair
./images/image_028.png, a man in a tuxedo standing in front of a table with a light
./images/image_034.png, a mosaic portrait of a woman wearing a headdress
./images/image_040.png, a group of spaceships flying through a space filled with stars
./images/image_046.png, a young man with black hair and a hoodie

```

## Usage

```bash
python inference.py --image_folder --output_folder
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
