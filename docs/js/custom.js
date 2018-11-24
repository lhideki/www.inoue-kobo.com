function display_thumbnail_card(el) {
    title = el.dataset.title;
    dir = el.dataset.dir;
    thumbnail = el.dataset.thumbnail;

    el.classList.add('col-md-4');

    card = document.createElement('div')
    card.classList.add('card');
    cardImg = document.createElement('img')
    cardImg.setAttribute('src', dir + '/' + thumbnail);
    cardBody = document.createElement('div');
    cardBody.classList.add('card-body');
    cardTitle = document.createElement('h5');
    cardTitle.classList.add('card-title');
    cardTitle.textContent = title;
    cardLink = document.createElement('a');
    cardLink.classList.add('btn', 'btn-primary');
    cardLink.setAttribute('href', dir + '/' + 'index.html');
    cardLink.textContent = 'Read';
    cardBody.appendChild(cardTitle);
    cardBody.appendChild(cardLink);
    
    card.appendChild(cardImg);
    card.appendChild(cardBody);

    el.appendChild(card);
}

window.addEventListener('load', function(event) {
    thumbnails = this.document.getElementsByClassName('thumbnail-card');
    for (el of thumbnails) {
        display_thumbnail_card(el);
    };
});