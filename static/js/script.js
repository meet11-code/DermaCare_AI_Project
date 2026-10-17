// ============================================================
// DERMACARE AI - GLOBAL JAVASCRIPT
// ============================================================

document.addEventListener("DOMContentLoaded", function () {

// ========================================================
// ELEMENTS
// ========================================================

const menuBtn =
    document.getElementById("menuBtn") ||
    document.getElementById("menu-btn");

const closeSidebarBtn =
    document.getElementById("closeSidebar") ||
    document.getElementById("close-btn");

const sidebar =
    document.getElementById("sidebar");

const imageInput =
    document.getElementById("imageInput");

const uploadBox =
    document.getElementById("uploadBox");

const fileName =
    document.getElementById("fileName");

const imagePreview =
    document.getElementById("imagePreview");

const previewImage =
    document.getElementById("previewImage");

const searchInput =
    document.getElementById("searchInput");


// ========================================================
// OPEN SIDEBAR
// ========================================================

if (menuBtn && sidebar) {

    menuBtn.addEventListener("click", function (event) {

        event.preventDefault();
        event.stopPropagation();

        sidebar.classList.add("active");

    });

}


// ========================================================
// CLOSE SIDEBAR
// ========================================================

if (closeSidebarBtn && sidebar) {

    closeSidebarBtn.addEventListener("click", function (event) {

        event.preventDefault();
        event.stopPropagation();

        sidebar.classList.remove("active");

    });

}


// ========================================================
// CLOSE SIDEBAR WHEN CLICKING OUTSIDE
// ========================================================

document.addEventListener("click", function (event) {

    if (!sidebar) {
        return;
    }

    if (!sidebar.classList.contains("active")) {
        return;
    }

    const clickedInsideSidebar =
        sidebar.contains(event.target);

    const clickedMenuButton =
        menuBtn &&
        menuBtn.contains(event.target);

    if (
        !clickedInsideSidebar &&
        !clickedMenuButton
    ) {

        sidebar.classList.remove("active");

    }

});


// ========================================================
// SIDEBAR LINKS
// ========================================================

const sidebarLinks = document.querySelectorAll("#sidebar a");

sidebarLinks.forEach(function (link) {

    link.addEventListener("click", function () {

        if (sidebar) {

            sidebar.classList.remove("active");

        }

    });

});


function handleSelectedImage(file) {

    if (!file) {
        return;
    }

    const allowedTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {

        alert("Only JPG, JPEG, PNG and WEBP images are allowed.");

        if (imageInput) {
            imageInput.value = "";
        }

        return;
    }


    // ---------------------------------------------
    // SHOW FILE NAME
    // ---------------------------------------------
    if (fileName) {

        fileName.textContent =
            "Selected: " + file.name;
    }


    // ---------------------------------------------
    // SHOW IMAGE PREVIEW
    // ---------------------------------------------
    if (previewImage && imagePreview) {

        const reader = new FileReader();

        reader.onload = function (event) {

            previewImage.src =
                event.target.result;

            imagePreview.style.display =
                "block";
        };

        reader.onerror = function () {

            console.error(
                "Unable to read selected image."
            );
        };

        reader.readAsDataURL(file);
    }
}


    // ========================================================
    // CHOOSE IMAGE
    // ========================================================

    if (imageInput) {

        imageInput.addEventListener(
            "change",
            function () {

                const file =
                    this.files[0];

                handleSelectedImage(file);

            }
        );

    }


  // =====================================================
// IMAGE UPLOAD - CLICK + DRAG & DROP
// =====================================================

if (uploadBox && imageInput) {

    // ---------------------------------------------
    // CLICK UPLOAD BOX
    // ---------------------------------------------
    uploadBox.addEventListener("click", function (event) {

        // Browse button / label ko double trigger mat karo
        if (event.target.closest(".browse-btn")) {
            return;
        }

        imageInput.click();
    });


    // ---------------------------------------------
    // FILE SELECTED
    // ---------------------------------------------
    imageInput.addEventListener("change", function () {

        const file = this.files && this.files[0];

        if (!file) {
            return;
        }

        handleSelectedImage(file);
    });


    // ---------------------------------------------
    // DRAG OVER
    // ---------------------------------------------
    uploadBox.addEventListener("dragover", function (event) {

        event.preventDefault();
        event.stopPropagation();

        uploadBox.classList.add("drag-over");
    });


    // ---------------------------------------------
    // DRAG ENTER
    // ---------------------------------------------
    uploadBox.addEventListener("dragenter", function (event) {

        event.preventDefault();
        event.stopPropagation();

        uploadBox.classList.add("drag-over");
    });


    // ---------------------------------------------
    // DRAG LEAVE
    // ---------------------------------------------
    uploadBox.addEventListener("dragleave", function (event) {

        event.preventDefault();
        event.stopPropagation();

        uploadBox.classList.remove("drag-over");
    });


    // ---------------------------------------------
    // DROP
    // ---------------------------------------------
    uploadBox.addEventListener("drop", function (event) {

        event.preventDefault();
        event.stopPropagation();

        uploadBox.classList.remove("drag-over");

        const files = event.dataTransfer.files;

        if (!files || files.length === 0) {
            return;
        }

        const file = files[0];

        const allowedTypes = [
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp"
        ];

        if (!allowedTypes.includes(file.type)) {

            alert("Only JPG, JPEG, PNG and WEBP images are allowed.");

            imageInput.value = "";

            return;
        }

        // File input ke andar dropped file set karo
        try {

            const dataTransfer = new DataTransfer();

            dataTransfer.items.add(file);

            imageInput.files = dataTransfer.files;

        } catch (error) {

            console.error("Unable to set dropped file:", error);
        }

        // Same preview function
        handleSelectedImage(file);
    });
}


    // ========================================================
    // SEARCH
    // ========================================================

    if (searchInput) {

        searchInput.addEventListener(
            "keyup",
            function () {

                const searchText =
                    searchInput.value
                    .toLowerCase()
                    .trim();


                const resultBlocks =
                    document.querySelectorAll(
                        ".result-block"
                    );


                resultBlocks.forEach(
                    function (block) {

                        const text =
                            block.innerText
                            .toLowerCase();


                        if (
                            searchText === "" ||
                            text.includes(searchText)
                        ) {

                            block.style.display =
                                "block";

                        } else {

                            block.style.display =
                                "none";

                        }

                    }
                );

            }
        );

    }


    // ========================================================
    // IMAGE MODAL - OPEN
    // ========================================================

    window.openImage =
        function (imageSrc) {

            const modal =
                document.getElementById(
                    "imageModal"
                );

            const modalImage =
                document.getElementById(
                    "modalImage"
                );


            if (
                !modal ||
                !modalImage
            ) {

                console.error(
                    "Image modal elements not found!"
                );

                return;

            }


            modalImage.src =
                imageSrc;

            modal.style.display =
                "flex";

        };


    // ========================================================
    // IMAGE MODAL - CLOSE
    // ========================================================

    window.closeImage =
        function () {

            const modal =
                document.getElementById(
                    "imageModal"
                );


            if (modal) {

                modal.style.display =
                    "none";

            }

        };


    // ========================================================
    // CLOSE MODAL OUTSIDE IMAGE
    // ========================================================

    document.addEventListener(
        "click",
        function (event) {

            const modal =
                document.getElementById(
                    "imageModal"
                );


            if (
                modal &&
                event.target === modal
            ) {

                window.closeImage();

            }

        }
    );


    // ========================================================
    // CLOSE MODAL WITH ESC
    // ========================================================

    document.addEventListener(
        "keydown",
        function (event) {

            if (
                event.key === "Escape"
            ) {

                window.closeImage();

            }

        }
    );

});


// ============================================================
// IMAGE + TEXT FORM VALIDATION
// ============================================================

document.addEventListener("DOMContentLoaded", function () {

    const uploadForm = document.getElementById("uploadForm");
    const imageInput = document.getElementById("imageInput");
    const problemText = document.getElementById("combinedProblemText");

    if (!uploadForm) {
        return;
    }

    uploadForm.addEventListener("submit", function (event) {

        const hasImage =
            imageInput &&
            imageInput.files &&
            imageInput.files.length > 0;

        const hasText =
            problemText &&
            problemText.value.trim().length > 0;

        // ====================================================
        // BOTH IMAGE AND TEXT ARE BLANK
        // ====================================================

        if (!hasImage && !hasText) {

            // Stop form from going to Flask
            event.preventDefault();

            // Remove previous validation message
            const oldError =
                uploadForm.querySelector(".form-validation-error");

            if (oldError) {
                oldError.remove();
            }

            // Create error message
            const errorMessage =
                document.createElement("div");

            errorMessage.className =
                "form-validation-error";

            errorMessage.innerHTML = `
                <i class="fa-solid fa-circle-exclamation"></i>
                Please select an image or describe your skin problem.
            `;

            // Put error just before analyze button
            const buttonBox =
                uploadForm.querySelector(
                    ".combined-analyze-button"
                );

            if (buttonBox) {
                buttonBox.before(errorMessage);
            } else {
                uploadForm.prepend(errorMessage);
            }

            // Scroll to Skin Detection section
            const skinSection =
                document.getElementById("skin-detection");

            if (skinSection) {
                skinSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });
            }

            return;
        }

        // ====================================================
        // VALID FORM
        // ====================================================

        // Remove validation error if it exists
        const oldError =
            uploadForm.querySelector(".form-validation-error");

        if (oldError) {
            oldError.remove();
        }

    });

});