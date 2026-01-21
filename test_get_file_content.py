from functions.get_file_content import get_file_content

content = get_file_content("calculator", "lorem.txt")

print(len(content))
print(content[content.find("[...File"):])

print(get_file_content("calculator", "main.py"))
print(get_file_content("calculator", "pkg/calculator.py"))
print(get_file_content("calculator", "/bin/cat"))
print(get_file_content("calculator", "pkg/does_not_exist.py"))