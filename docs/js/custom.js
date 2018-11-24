function show_thumbnail_card(el) {
    title = el.dataset.title;
    dir = el.dataset.dir;
    thumbnail = el.dataset.thumbnail;

    template = document.importNode(document.querySelector('#card-template').content, true);
    template.querySelector('img').setAttribute('src', dir + '/' + thumbnail);
    template.querySelector('h5').textContent = title;
    template.querySelector('a').setAttribute('href', dir + '/index.html');

    el.classList.add('col-md-4');
    el.appendChild(template);
}

window.addEventListener('load', function(event) {
    thumbnails = this.document.getElementsByClassName('thumbnail-card');
    for (el of thumbnails) {
        show_thumbnail_card(el);
    };
});