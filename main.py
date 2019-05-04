import glob
import os
import sys
import re

COL_NUM = 3
SUMMARY_SIZE = 96

def get_thumbnail_element(dir, index_filename='index.md', thumbnail_filename='images/thumbnail.png'):
    html = '<div class="container-fluid">\n'
    filenames = glob.glob(f'{dir}/*/{index_filename}')
    basedir = os.path.basename(dir)
    for i, filename in enumerate(filenames):
        title = read_title(filename)
        description = read_description(filename)
        content_dir = os.path.join(basedir, os.path.basename(os.path.dirname(filename)))
        thumbnail = os.path.join(content_dir, thumbnail_filename)

        if i == 0  or i % COL_NUM == 0:
            html += '  <div class="row" style="margin-bottom: 1.5em">\n'
        html += '    <div class="col-md-4">\n'
        card = get_card_element(title, content_dir, thumbnail, description)
        html += card
        html += '    </div>\n'

        if i > 0 and (i + 1) % COL_NUM == 0:
          html += '  </div>\n'
        elif i == len(filenames) - 1:
          html += '  </div>\n'
    html += '</div>'

    return html


def read_title(filename):
    with open(filename) as f:
        title = re.search('\# (.+)', f.readlines()[0]).group(1)
    return title

def read_description(filename):
    description = ''

    with open(filename) as f:
        for line in f.readlines():
            if re.search('^\#+\s.+', line):
                continue
            description += re.sub(r'\[(.+?)\]\(.+?\)', r'`\1`', line)
            if len(description) > SUMMARY_SIZE:
                description = description[:SUMMARY_SIZE]
                description += '・・・'
                break

    description.replace('\n', '<br />')

    return description

def get_card_element(title, dir, thumbnail, description):
    card = ''
    card += f'    <div class="card">\n'
    card += f'      <div class="card-thumbnail">\n'
    card += f'        <img src="{thumbnail}" />\n'
    card += f'      </div>\n'
    card += f'      <div class="card-body">\n'
    card += f'        <h5 class="card-title">{title}</h5>\n'
    card += f'        <p>{description}</p>\n'
    card += f'      </div>\n'
    card += f'      <div class="actions">\n'
    card += f'        <a class="btn btn-primary" href="{dir}/index.html">Read</a>\n'
    card += f'      </div>'
    card += f'    </div>\n'

    return card


def declare_variables(variables, macro):
    @macro
    def print_thumbnail(dir):
      return get_thumbnail_element(dir)

if __name__ == '__main__':
    dir = sys.argv[1]
    element = get_thumbnail_element(dir)

    print(element)
