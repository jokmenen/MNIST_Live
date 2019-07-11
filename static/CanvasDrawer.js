
        $(document).ready(function(){
            var clic=false;
            var xCoord,yCoord="";
            var canvas=document.getElementById("DigitCanvas");
            var cntx=canvas.getContext("2d");
            cntx.strokeStyle="black";
            cntx.lineWidth=4;
            cntx.lineCap="round";
            cntx.fillStyle="#fff";
            cntx.fillRect(0,0,canvas.width,canvas.height);

            $("#DigitCanvas").mousedown(function(canvas){
                var xy = getCursorPosition(this,canvas)
                var scale = elementScale(canvas);
                clic=true;
                cntx.save();
                xCoord=xy[0];
                yCoord=xy[1];
                /"console.log(xCoord , yCoord)"/
            });

            $(document).mouseup(function(){
                clic=false
            });

            $(document).click(function(){
                clic=false
            });

            function elementScale(el) {
                var scalex = el.offsetWidth === 0 ? 0 : (el.width / el.offsetWidth);
                var scaley = el.offsetHeight === 0 ? 0 : (el.height / el.offsetHeight);
                return [scalex,scaley]

            }

            $("#DigitCanvas").mousemove(function(canvas){
                if(clic==true){
                    var xy = getCursorPosition(this,canvas)
                    var scale = elementScale(this)
                    xCoord=xy[0]* scale[0] ;
                    yCoord=xy[1] * scale[1];
                    console.log(xy, scale)
                    cntx.beginPath();
                    cntx.moveTo(xCoord,yCoord);
                    cntx.lineTo(xCoord,yCoord);
                    cntx.stroke();
                    cntx.closePath();

                }
            });

            $("#reset").click(function(){
                cntx.fillStyle="#fff";
                cntx.fillRect(0,0,canvas.width, canvas.height);
                cntx.strokeStyle="black";
                cntx.fillStyle="black"
            });

            $("#DigitCanvas").mouseup(function(e){
                var image = new Image();
                image.id = "pic";
                image.src = canvas.toDataURL();
                var datastr = canvas.toDataURL();
                console.log(image)
                // Send image to python for transforming
                // maybe myurl.com/sendPic with the dataurl in header
                // unpack and resize in python
                // document.getElementById('image_for_crop').appendChild(image);


                $.ajax({
                    url: "/sendImage",
                    type: "POST",
                    data: datastr,
                    success: function (prediction) {
                      $('#Predict_L').text(prediction);
                //         alert("Hoi")
                    },
                    cache: false,
                    contentType: "text; charset=utf-8",
                  });

                 e.preventDefault();

                // $.post("/sendImage",
                //     {
                //         data: image,
                //     },
                //     function(prediction){
                //     // $('#Predict_L').innerText = prediction;
                //         alert("Hoi")
                // });

            })
        })

        function getCursorPosition(canvas, event) {
    var rect = canvas.getBoundingClientRect();
    var x = event.clientX - rect.left;
    var y = event.clientY - rect.top;
    return [x,y];
}
