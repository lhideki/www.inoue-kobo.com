import glob
import os
import sys
import re

def get_thumbnail_element(dir, index_filename='index.md', thumbnail_filename='images/thumbnail.png'):
    html = '<div class="container-fluid">\n'
    filenames = glob.glob(f'{dir}/*/{index_filename}')
    basedir = os.path.basename(dir)
    for i, filename in enumerate(filenames):
        title = read_title(filename)
        content_dir = os.path.join(basedir, os.path.basename(os.path.dirname(filename)))
        thumbnail = os.path.join(content_dir, thumbnail_filename)

        if i % 3 == 0:
            html += '  <div class="row">\n'
        html += '    <div class="col-md-4">\n'
        card = get_card_element(title, content_dir, thumbnail)
        html += card
        html += '    </div>\n'

        if i > 0 and i % 3 == 0:
          html += '  </div>\n'
        elif i == len(filenames) - 1:
          html += '  </div>\n'
    html += '</div>'

    return html


def read_title(filename):
    with open(filename) as f:
        title = re.search('\# (.+)', f.readlines()[0]).group(1)
    return title


def get_card_element(title, dir, thumbnail):
    card = ''
    card += f'    <div class="card">\n'
    card += f'      <img src="{thumbnail}" />\n'
    card += f'      <div class="card-body">\n'
    card += f'        <h5 class="card-title">{title}</h5>\n'
    card += f'        <a class="btn btn-primary" href="{dir}/index.html">Read</a>\n'
    card += f'      </div>\n'
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
