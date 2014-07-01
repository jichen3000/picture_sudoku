function p(msg){
  $('p#messages').text(msg);
}

Object.size = function(obj) {
  var size = 0, key;
  for (key in obj) {
    if (obj.hasOwnProperty(key)) size++;
  }
  return size;
};

$(function() {
  $(".cell").hover(
    function() {
      $(this).addClass("hover");
    },
    function(){
      $(this).removeClass("hover");
    }
  );
  var fixedClass = 'fixed';
  var answeredClass = 'answered';
  function displayPoints(points, needClass){
    for (var key in points){
      $("#"+key).text(points[key]).addClass(needClass);
    }
  }
  function clear(){
    // p('clear');
    $('#progress').hide();
    $('canvas').remove();
    $('#sudoku-99').find('.cell').text('').removeClass(answeredClass).removeClass(fixedClass);    
  }
  function buttonLoading(jqueryButton){
    jqueryButton.prop("disabled", true);
    jqueryButton.attr("data-init-text",jqueryButton.text());
    // If I just use jqueryButton.text("123") or I just use jqueryButton.button('loading'), it will report a error:
    // text cannot call methods on fileupload prior to initialization; attempted to call method 'process'
    jqueryButton.children("span").text(jqueryButton.attr("data-loading-text"));
  }
  function buttonStepping(jqueryButton){
    jqueryButton.children("span").text(jqueryButton.attr("data-stepping-text"));
  }
  function buttonReset(jqueryButton){
    jqueryButton.prop("disabled", false);
    jqueryButton.children("span").text(jqueryButton.attr("data-init-text"));
  }
  $('#fileupload').fileupload({
    type: 'POST',
    url:  '/sudoku/image/result',
    dataType: 'json',
    acceptFileTypes: /^image\/(gif|jpeg|png)$/,
    maxFileSize: 10000000, // 10MB
    // check if the browser support
    disableImageResize: /Android(?!.*Chrome)|Opera/
        .test(window.navigator && navigator.userAgent),
    imageMaxWidth: 960,
    imageMaxHeight: 960,
    // imageCrop: true, // Force cropped images
    imageForceResize: true, // resize images
    previewMaxWidth: 640,
    previewMaxHeight: 960,
    done: function (event, data) {
      $('#progress').hide();
      // $("#upload").button('reset');
      buttonReset($("#upload"));
      // the data has been changed to be json object, so don't need to JSON.parse
      if(data.result['status'] == 'SUCCESS'){
        displayPoints(data.result[fixedClass], fixedClass);
        displayPoints(data.result[answeredClass], answeredClass);
        p('Done! '+data.result['pic_file_name']);
      }else{
        if(data.result[fixedClass]){
          displayPoints(data.result[fixedClass], fixedClass);
        }
        p('Sorry, this puzzle cannot be answered! The file name is'+data.result['pic_file_name']);

      }
    },
    processdone:function (event, data) {
      // append_p('processdone');
      currentFile = data.files[data.index];
      // append_p(currentFile.name);
      // append_p(currentFile.type);
      // append_p(currentFile.size);
      $("section#preview").append(currentFile.preview);
    },
    processfail:function (event, data) {
      p(data.files[data.index].error);
    },
    progressall: function (event, data) {
      var progress = parseInt(data.loaded / data.total * 100, 10);
      // for upload process, it's only 50% maximumly.
      progress = 0.5 * progress
      if (progress >= 50){
        buttonStepping($("#upload"));
      }
      // p(progress);
      $('#progress').show();
      $('#progress .progress-bar').css(
          'width',
          progress + '%'
      ).text(progress + '%').attr('aria-valuenow',progress);
    }
  }).on('fileuploadadd', function (event, data) {
    // use this way to get add event, will not stop the upload automatically.
    // $("#upload").button('loading');
    buttonLoading($("#upload"));
    clear();
  });

});
