import zipfile


from ofact.settings import ROOT_PATH

file_directory_path_as_str = str(ROOT_PATH) + "/interfaces/frontend/scenario_data/"

files = ["orders.xlsx", "products.xlsx", "processes.xlsx", "resources.xlsx", "resource_utilization.xlsx",
             "scenario_descriptions.xlsx"]
zip_file_path = 'scenario_data/scenario_comparison.zip'
with zipfile.ZipFile(zip_file_path, "w") as zip_file:
    for file in files:
        zip_file.write(file_directory_path_as_str + file, arcname=file)