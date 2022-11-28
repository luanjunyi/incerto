const MAX = 500;

function scroll_and_sleep(id) {
  if (id >= MAX) {
    return;
  }
  window.scrollTo(0, document.documentElement.scrollHeight);
  console.log("scrolled with ID=" + id.toString());
  setTimeout(() => {
    scroll_and_sleep(id+1);
  }, "1000");
}
