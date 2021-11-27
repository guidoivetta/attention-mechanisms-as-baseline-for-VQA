import glob
# All files ending with .txt
data_files_paths = glob.glob("./*.txt")

diseaseNameOrder = []
diseaseNameIdEnd = [0]


print("{\n\"images\": [\n")
i=0
for data_file_path in data_files_paths:
  # print(data_file_path,i)
  with open(data_file_path) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

  for image_path in lines:
    image_item = "{\n"f'  "file_name": "{image_path}",\n'+f'  "height": 256,\n  "width": 256,\n'f'  "id": {i}\n'"},"
    print(image_item)
    i+=1
  diseaseNameOrder.append(data_file_path)
  diseaseNameIdEnd.append(i)
print("],")

print("\"annotations\": [\n")
i=0
for j in range(len(diseaseNameIdEnd)-1):
  (plant_name, _, diseaseName) = diseaseNameOrder[j][len('./'):][:-len('.txt')].partition("__")
  diseaseName = diseaseName.replace("_", " ")
  # print(plant_name,diseaseName)
  for image_id in range(diseaseNameIdEnd[j],diseaseNameIdEnd[j+1]):
    sampleDescriptions = []
    if(diseaseName == "healthy"):
      sampleDescriptions.append(f"Un {plant_name} saludable.")
      sampleDescriptions.append(f"Una hoja de {plant_name} en buena salud.")
      sampleDescriptions.append(f"Una hoja de {plant_name} en buenas condiciones.")
      sampleDescriptions.append(f"Una saludable hoja de {plant_name}.")
      sampleDescriptions.append(f"Una hoja de {plant_name} sin enfermedades.")
      sampleDescriptions.append(f"Una hoja de {plant_name} perfectamente saludable.")
    else:
      sampleDescriptions.append(f"Una hoja de {plant_name} con {diseaseName}.")
      sampleDescriptions.append(f"Una hoja de {plant_name} en malas condiciones, tiene {diseaseName}.")
      sampleDescriptions.append(f"Una hoja de {plant_name} en mala salud, tiene {diseaseName}.")
      sampleDescriptions.append(f"Un {plant_name} enfermo con {diseaseName}")
    for description in sampleDescriptions:
      annotation_item = "{\n"f'  "image_id": {image_id},\n'+f'  "id": {i},\n'f'  "caption": "{description}"\n'"},"
      print(annotation_item)
      i+=1
print("]\n}")
