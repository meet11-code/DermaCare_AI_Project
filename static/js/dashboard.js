// =====================================================
// DERMACARE AI - IMAGE UPLOAD + DRAG & DROP + PREVIEW
// =====================================================

document.addEventListener("DOMContentLoaded", function () {

    const uploadForm = document.getElementById("uploadForm");
    const uploadBox = document.getElementById("uploadBox");
    const imageInput = document.getElementById("imageInput");
    const imagePreview = document.getElementById("imagePreview");
    const previewImage = document.getElementById("previewImage");
    const fileName = document.getElementById("fileName");


    // =====================================================
    // CHECK ELEMENTS
    // =====================================================

    if (!uploadBox || !imageInput) {
        console.log("Upload elements not found.");
        return;
    }


    // =====================================================
    // ALLOWED IMAGE TYPES
    // =====================================================

    const allowedTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    ];


    // =====================================================
    // SHOW IMAGE PREVIEW
    // =====================================================

    function showImage(file) {

        if (!file) {
            return;
        }


        // Check file type

        if (!allowedTypes.includes(file.type)) {

            alert(
                "Please select a valid JPG, JPEG, PNG or WEBP image."
            );

            imageInput.value = "";

            if (imagePreview) {
                imagePreview.style.display = "none";
            }

            if (fileName) {
                fileName.textContent = "";
            }

            return;
        }


        // Check file size
        // Maximum 10 MB

        if (file.size > 10 * 1024 * 1024) {

            alert(
                "Image size must be less than 10 MB."
            );

            imageInput.value = "";

            if (imagePreview) {
                imagePreview.style.display = "none";
            }

            if (fileName) {
                fileName.textContent = "";
            }

            return;
        }


        // =================================================
        // SET FILE NAME
        // =================================================

        if (fileName) {

            fileName.textContent =
                "Selected: " + file.name;

        }


        // =================================================
        // CREATE IMAGE PREVIEW
        // =================================================

        const reader = new FileReader();


        reader.onload = function (event) {

            if (previewImage) {

                previewImage.src =
                    event.target.result;

            }


            if (imagePreview) {

                imagePreview.style.display =
                    "block";

            }

        };


        reader.onerror = function () {

            alert(
                "Unable to preview this image."
            );

        };


        reader.readAsDataURL(file);

    }



    // =====================================================
    // CHOOSE IMAGE
    // =====================================================

    imageInput.addEventListener(
        "change",
        function (event) {

            const file =
                event.target.files[0];

            showImage(file);

        }
    );



    // =====================================================
    // DRAG OVER
    // =====================================================

    uploadBox.addEventListener(
        "dragover",
        function (event) {

            // Prevent browser default
            event.preventDefault();

            event.stopPropagation();


            uploadBox.classList.add(
                "drag-over"
            );

        }
    );



    // =====================================================
    // DRAG LEAVE
    // =====================================================

    uploadBox.addEventListener(
        "dragleave",
        function (event) {

            event.preventDefault();

            event.stopPropagation();


            uploadBox.classList.remove(
                "drag-over"
            );

        }
    );



    // =====================================================
    // DROP IMAGE
    // =====================================================

    uploadBox.addEventListener(
        "drop",
        function (event) {

            // VERY IMPORTANT
            // Stop browser from opening image

            event.preventDefault();

            event.stopPropagation();


            uploadBox.classList.remove(
                "drag-over"
            );


            // Get dropped files

            const files =
                event.dataTransfer.files;


            if (
                !files ||
                files.length === 0
            ) {

                return;

            }


            const file =
                files[0];


            // Check image

            if (
                !allowedTypes.includes(
                    file.type
                )
            ) {

                alert(
                    "Please drop a JPG, JPEG, PNG or WEBP image."
                );

                return;

            }


            // =================================================
            // PUT DROPPED FILE INTO FILE INPUT
            // =================================================

            try {

                const dataTransfer =
                    new DataTransfer();


                dataTransfer.items.add(
                    file
                );


                imageInput.files =
                    dataTransfer.files;

            }

            catch (error) {

                console.error(
                    "Could not set dropped file:",
                    error
                );

            }


            // =================================================
            // SHOW PREVIEW
            // =================================================

            showImage(file);

        }
    );



    // =====================================================
    // PREVENT BROWSER FROM OPENING DROPPED IMAGE
    // =====================================================

    document.addEventListener(
        "dragover",
        function (event) {

            event.preventDefault();

        },
        false
    );


    document.addEventListener(
        "drop",
        function (event) {

            // If drop is outside upload box,
            // prevent browser opening file

            if (
                !uploadBox.contains(
                    event.target
                )
            ) {

                event.preventDefault();

                event.stopPropagation();

            }

        },
        false
    );



    // =====================================================
    // FORM SUBMIT VALIDATION
    // =====================================================

    if (uploadForm) {

        uploadForm.addEventListener(
            "submit",
            function (event) {

                if (
                    !imageInput.files ||
                    imageInput.files.length === 0
                ) {

                    event.preventDefault();


                    alert(
                        "Please choose or drag an image first."
                    );


                    return false;

                }

            }
        );

    }


    // =====================================================
    // DEBUG
    // =====================================================

    console.log(
        "DermaCare AI Dashboard Image Upload Ready"
    );

});