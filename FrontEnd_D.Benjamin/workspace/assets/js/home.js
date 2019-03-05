
%if ((stash 'id') ne "") {
  window.onload = function(){

    function DownloadImg() {
      if (confirm("Do you want to download the good images? <%= stash 'results' %> of the images were good.")) {
        window.href.location = "/download/<%= stash 'id' %>"

      } else {

      }
    }
  }
%}
