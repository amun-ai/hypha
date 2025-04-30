import { WebUI } from "jsr:@webui/deno-webui";

const myWindow = new WebUI();
await myWindow.show(
  '<html><script src="webui.js"></script> Hello World! </html>',
);
await WebUI.wait();
