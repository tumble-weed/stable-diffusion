

import gdown

#gdrive_folder_link = "https://drive.google.com/drive/folders/18ngh0sEPurY3PfS3KlxSEF3TcYzzXgAc?usp=share_link"
gdrive_folder_link = "https://drive.google.com/drive/folders/1k0CUM03n31uIuk-xxpFDukHLfBbj4l1J?usp=sharing"

gdown.download_folder(gdrive_folder_link, quiet=False)
