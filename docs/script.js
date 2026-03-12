function openTab(evt, tabName) {
  var i, tabcontent, tablinks;
  tabcontent = document.getElementsByClassName("tab-content");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
    tabcontent[i].classList.remove("active");
  }
  tablinks = document.getElementsByClassName("tab-btn");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(tabName).style.display = "block";
  // Slight delay to re-trigger animation
  setTimeout(() => {
    document.getElementById(tabName).classList.add("active");
  }, 10);
  evt.currentTarget.className += " active";
}

function openSubTab(evt, tabName) {
  var i, tabcontent, tablinks;
  // Look only inside the currently active parent tab-content if needed, 
  // but unique IDs allow us to just hide all sub-tabs.
  tabcontent = document.getElementsByClassName("sub-tab-content");
  for (i = 0; i < tabcontent.length; i++) {
    tabcontent[i].style.display = "none";
    tabcontent[i].classList.remove("active");
  }
  tablinks = document.getElementsByClassName("sub-tab-btn");
  for (i = 0; i < tablinks.length; i++) {
    tablinks[i].className = tablinks[i].className.replace(" active", "");
  }
  document.getElementById(tabName).style.display = "block";
  setTimeout(() => {
    document.getElementById(tabName).classList.add("active");
  }, 10);
  evt.currentTarget.className += " active";
}

document.addEventListener("DOMContentLoaded", function() {
  document.getElementById("audio-interventions").style.display = "block";
  document.getElementById("cot-filler").style.display = "block";
});
