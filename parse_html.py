import os
log_path = "/Users/manish/.gemini/antigravity/brain/a29d5b1b-0826-44e1-b2ea-116cb09513f1/.system_generated/logs/overview.txt"
with open(log_path, 'r') as f:
    lines = f.readlines()

html_lines = []
in_html = False
for line in lines:
    if line.strip() == "+<!DOCTYPE html>":
        in_html = True
    
    if in_html:
        if line.startswith("+"):
            html_lines.append(line[1:]) # Drop leading +
        elif line.startswith("<!DOCTYPE html>"):
            html_lines.append(line)
        else:
            # Reached end of diff
            if "</html>" in line:
                html_lines.append(line.lstrip("+"))
                break

if html_lines:
    with open("/Users/manish/search_benchmark /index.html", "w") as out:
        out.writelines(html_lines)
    print(f"Successfully wrote {len(html_lines)} lines to index.html")
else:
    print("Could not find HTML in logs")
