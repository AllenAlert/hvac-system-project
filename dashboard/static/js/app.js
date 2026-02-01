// HVAC Dashboard - global behavior
document.addEventListener('DOMContentLoaded', function () {
  // Card links (overview)
  document.querySelectorAll('.card-link[data-href]').forEach(function (card) {
    card.addEventListener('click', function () {
      const href = card.getAttribute('data-href');
      if (href) window.location.href = href;
    });
  });
});
