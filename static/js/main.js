$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#file-error').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        var file = this.files && this.files[0];
        var allowedTypes = ['image/jpeg', 'image/png'];
        var maxSize = 10 * 1024 * 1024;
        $('#file-error').hide().text('');
        $('#result').hide().text('');

        if (!file) {
            $('.image-section').hide();
            return;
        }

        if (allowedTypes.indexOf(file.type) === -1 || file.size > maxSize) {
            $('.image-section').hide();
            $('#file-error').text(
                file.size > maxSize
                    ? 'This image is larger than 10 MB.'
                    : 'Please choose a JPG or PNG image.'
            ).show();
            this.value = '';
            return;
        }

        $('.image-section').show();
        $('#btn-predict').prop('disabled', false).show();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).prop('disabled', true).text('Predicting...');
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result').text(data);
                $('#btn-predict').prop('disabled', false).text('Predict image');
                console.log('Success!');
            },
            error: function (xhr) {
                $('.loader').hide();
                $('#btn-predict').prop('disabled', false).text('Try again');
                $('#result').fadeIn(600);
                var message = 'Prediction failed. Please try another image.';
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    message = xhr.responseJSON.error;
                }
                $('#result').text(message);
            }
        });
    });

});