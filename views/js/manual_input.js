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
  var duplicatedClass = 'duplicated';
  var updateElement = null;

  $('#sudoku-99').find('.cell').click(function(e){
    var position = { top: e.pageY+5, left: e.pageX+5 }
    $('#choose-value-dialog').modal({backdrop: false});
    $('#choose-value-dialog').offset(position);
    // p(e.pageX+":"+e.pageY);
    updateElement = this;
  });
  
  $('#choose-value-dialog').find('.cell').click(function(e){
    // p(this.innerText);
    $(updateElement).addClass(fixedClass).text(this.innerText);
    $('#choose-value-dialog').modal('hide');
    refreshFixedPointsCount();
  });
  
  $('#null').click(function(){
    clearCell(updateElement);
    $('#choose-value-dialog').modal('hide');
    refreshFixedPointsCount();
  });
  function clearCell(current){
    $(current).text('').removeClass(answeredClass).removeClass(fixedClass).removeClass(duplicatedClass);
  }
  function getFixedPoints(){
    var fixedPoints = {};
    $('.'+fixedClass).each(function(){
      fixedPoints[this.id] = Number(this.innerText);
    });
    return fixedPoints;    
  }

  function computeRagionIndex(rowIndex, colIndex){
    return Math.floor(rowIndex / 3) * 3 + Math.floor(colIndex / 3);
  }
  function validateAll(fixedPoints){
    if (!validateFixedPointsCount(fixedPoints)){ 
      p("No sufficient set values! At least 17 values!");
      return false; 
    }
    if (!validateAnswered()){ 
      p("The answer has already been computed!");
      return false; 
    }
    var duplicatedValues = validatePointsDuplicated(fixedPoints);
    if (duplicatedValues!=true){ 
      p("There some duplicated values in the cells!");
      $("#"+duplicatedValues['previousKey']).addClass(duplicatedClass);
      $("#"+duplicatedValues['currentKey']).addClass(duplicatedClass);
      return false; 
    }
    return true; 
  }
  function validateFixedPointsCount(fixedPoints){
    return (Object.size(fixedPoints) >= 17);
  }
  function validateAnswered(){
    var pointsCount = $('.'+answeredClass).length;
    return (pointsCount == 0);  
  }
  function validatePointsDuplicated(points){
    var shownNumbers = {};
    for (var key in points){
      var indexs = key.split('_');
      var numberValue = points[key]
      var curNumberRowIndex = 'Row'+indexs[0]+numberValue;
      var curNumberColIndex = 'Col'+indexs[1]+numberValue;
      var curNumberRagionIndex = 'Region'+computeRagionIndex(indexs[0],indexs[1])+numberValue;
      if (shownNumbers[curNumberRowIndex] != undefined){
        return {'previousKey':shownNumbers[curNumberRowIndex],'currentKey':key,'numberValue':numberValue};
      }
      shownNumbers[curNumberRowIndex] = key;
      if (shownNumbers[curNumberColIndex] != undefined){
        return {'previousKey':shownNumbers[curNumberColIndex],'currentKey':key,'numberValue':numberValue};
      }
      shownNumbers[curNumberColIndex] = key;
      if (shownNumbers[curNumberRagionIndex] != undefined){
        return {'previousKey':shownNumbers[curNumberRagionIndex],'currentKey':key,'numberValue':numberValue};
      }
      shownNumbers[curNumberRagionIndex] = key;
    }
    return true; 
  }
  function refreshFixedPointsCount(){
    var pointsCount = $('.'+fixedClass).length+$('.'+answeredClass).length;
    $(".values-count").text("values count: "+pointsCount);
  }
  function displayPoints(points, needClass){
    for (var key in points){
      $("#"+key).text(points[key]).addClass(needClass);
    }
  }
  $('button#answer').click(function(){
    var curButton = $(this)
    var fixedPoints = getFixedPoints();
    if (!validateAll(fixedPoints)){ 
      return; 
    }
    curButton.button('loading');
    $.ajax({
      type: 'POST',
      url: '/sudoku/input/result',
      data: JSON.stringify(fixedPoints),
      contentType: 'application/json',
      success: function(result){
        //p('get success!'+result);
        if (result==="false"){
          curButton.button('reset');
          p('Sorry! This quiz has not an answer at all!');
          return false;
        }      
        displayPoints(JSON.parse(result), answeredClass);
        curButton.button('reset');
        p('Successful!');
        refreshFixedPointsCount();
      }
    });
    //p(JSON.stringify(fixedPoints));
    // $.get('/sudoku/sudokuresult',{fix_values:JSON.stringify(fixedPoints)},function(result){
    //   //p('get success!'+result);
    //   if (result==="false"){
    //     $('#process-dialog').modal('hide');
    //     p('Sorry! This quiz has not an answer at all!');
    //     return false;
    //   }      
    //   displayPoints(JSON.parse(result), answeredClass);
    //   $('#process-dialog').modal('hide');
    //   p('Successful!');
    //   refreshFixedPointsCount();
    // }, "json");
  });
  $('button#clear').click(function(){
    $('.'+answeredClass).each(function(){
      clearCell(this);
    });
    $('.'+fixedClass).each(function(){
      clearCell(this);
    });
    refreshFixedPointsCount();
    p("Cleared all values in the cells.");
  });
  $('button#record').click(function(){
    if (!validateFixedPoints()){
      p("No fixed values!");
      return; 
    }
    var fixedPoints = getFixedPoints();
    p(JSON.stringify(fixedPoints));
  });
  $('button#easy-example').click(function(){
    $('button#clear').click();
    var fixedPoints = '{"0_0":5,"1_0":3,"0_1":6,"1_2":9,"2_2":8,' +
                      '"4_0":7,"3_1":1,"4_1":9,"5_1":5,"7_2":6,' +
                      '"0_3":8,"0_4":4,"0_5":7,"4_3":6,"3_4":8,' +
                      '"5_4":3,"4_5":2,"8_3":3,"8_4":1,"8_5":6,' +
                      '"1_6":6,"3_7":4,"4_7":1,"5_7":9,"4_8":8,' +
                      '"6_6":2,"7_6":8,"8_7":5,"7_8":7,"8_8":9}';
    displayPoints(JSON.parse(fixedPoints), fixedClass);
    refreshFixedPointsCount();
    p("Added values in the cells as an example, then you just click the Answer button.");
  });
  $('button#difficaul-example').click(function(){
    $('button#clear').click();
    var fixedPoints = '{"0_0": 9, "0_8": 5, "1_1": 4, "1_3": 3, "1_7": 2,' +
                      ' "2_2": 8, "2_6": 1, "3_1": 7, "3_3": 6, "3_5": 3, ' +
                      ' "4_4": 8, "5_3": 7, "5_5": 9, "5_7": 6, "6_2": 1, ' +
                      ' "6_6": 9, "7_1": 3, "7_5": 6, "7_7": 4, "8_0": 5, ' +
                      ' "8_8": 8}';
    displayPoints(JSON.parse(fixedPoints), fixedClass);
    refreshFixedPointsCount();
    p("Added values in the cells as an example, then you just click the Answer button.");
  });
  
  refreshFixedPointsCount();
});
