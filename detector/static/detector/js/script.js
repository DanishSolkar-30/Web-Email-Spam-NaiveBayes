const messageInput = document.getElementById("message");
const charCount = document.getElementById("charCount");
const clearButton = document.getElementById("clearButton");
const emailForm = document.getElementById("emailForm");
const analyzeButton = document.getElementById("analyzeButton");
const buttonText = document.getElementById("buttonText");


/* ================= CHARACTER COUNTER ================= */

if (messageInput && charCount) {

    function updateCharacterCount() {

        charCount.textContent =
            messageInput.value.length;

    }

    messageInput.addEventListener(
        "input",
        updateCharacterCount
    );

    updateCharacterCount();
}


/* ================= CLEAR BUTTON ================= */

if (clearButton && messageInput) {

    clearButton.addEventListener("click", function () {

        messageInput.value = "";

        if (charCount) {
            charCount.textContent = "0";
        }

        messageInput.focus();

    });

}


/* ================= FORM SUBMISSION ================= */

if (emailForm && analyzeButton) {

    emailForm.addEventListener("submit", function () {

        analyzeButton.disabled = true;

        buttonText.textContent = "Analyzing...";

    });

}