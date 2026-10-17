// ============================================================
// STEP 8.6 - IMAGE PREVIEW
// ============================================================

function openImage(imageSrc) {

    const modal = document.getElementById("imageModal");

    const modalImage = document.getElementById("modalImage");

    modalImage.src = imageSrc;

    modal.style.display = "flex";
}


// ============================================================
// CLOSE IMAGE
// ============================================================

function closeImage() {

    const modal = document.getElementById("imageModal");

    modal.style.display = "none";

}


// ============================================================
// CLOSE WHEN CLICKING OUTSIDE IMAGE
// ============================================================

document.addEventListener("click", function(event) {

    const modal = document.getElementById("imageModal");

    if (event.target === modal) {

        closeImage();

    }

});


// ============================================================
// CLOSE WITH ESC KEY
// ============================================================

document.addEventListener("keydown", function(event) {

    if (event.key === "Escape") {

        closeImage();

    }

});