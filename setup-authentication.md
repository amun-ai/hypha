## Setup Authentication

Internally, Hypha uses auth0 to manage authentication. This allows us to use a variety of authentication providers, including Google, GitHub.

The default setting in hypha uses common auth0 setting managed by us, but you can also setup your own auth0 account and use it.

### Setup Auth0 Authentication

To set up your own account, follow these steps:
 - go to https://auth0.com/ and create an account, or re-use an existing Github or Google Account.
 - For the first time, you will be asked to create a "Tenant Domain" and choose a "Region", choose any name for the domain (e.g. hypha), and choose a suitable for the region (e.g. US or EU). Then click "Create".
 - After that you should be logged in to the auth0 dashboard. Click on "Applications" on the left menu, and then click on "Create Application".
 - Give your application a name (e.g. hypha), and choose "Single Page Web Applications" as the application type. Then click "Create".
 - Now go to the "Settings" tab of your application, and copy the "Domain" and "Client ID" values to create environment variables for running Hypha:
 ```
 AUTH0_CLIENT_ID=paEagfNXPBVw8Ss80U5RAmAV4pjCPsD2 # replace with your own value from the "Settings" tab
 AUTH0_DOMAIN=amun-ai.eu.auth0.com # replace with your own value from the "Settings" tab
 AUTH0_AUDIENCE=https://amun-ai.eu.auth0.com/api/v2/ # replace 'amun-ai.eu.auth0.com' to your own auth0 domain
 AUTH0_ISSUER=https://amun.ai/ # keep it or replace 'amun.ai' to any website you want to use as the issuer
 AUTH0_NAMESPACE=https://amun.ai/ # keep it or replace 'amun.ai' to any identifier you want to use as the namespace
 ```
 
 You can either set the environment variables in your system, or create a `.env` file in the root directory of Hypha, and add the above lines to the file.
 - Importantly, you also need to configure your own hypha server domain so Auth0 will allow it to login from your own domain. 
 For example, if you want to serve hypha server at https://my-org.com, you need to set the following in "Settings" tab:
    * scroll down to the "Allowed Callback URLs" section, and add the following URLs: https://my-org.com
    * scroll down to the "Allowed Logout URLs" section, and add the following URLs: https://my-org.com/public/apps/hypha-login/
    * scroll down to the "Allowed Web Origins" section, and add the following URLs: https://my-org.com
    * scroll down to the "Allowed Origins (CORS)" section, and add the following URLs: https://my-org.com
 For local development, you can also add `http://127.0.0.1:9527` to the above URLs, separated by comma. For example, "Allowed Callback URLs" can be `https://my-org.com,http://http://127.0.0.1:9527`.
 - Now you can start the hypha server (with the AUTH0 environment variables, via `python3 -m hypha.server --host=0.0.0.0 --port=9527`), and you should be able to test it by going to https://my-org.com/public/apps/hypha-login/ (replace with your own domain) or http://127.0.0.1:9527/public/apps/hypha-login.
 - By default, auth0 will provide a basic username-password-authentication which will store user information at auth0. You can also add other authentication providers (e.g. Google, Github) in the "Authenticaiton" tab of your application in Auth0 dashboard.
    * In order to add Google, click "Social", click "Create Connection", find Google/Gmail, and click "Continue", you will need to obtain the Client ID by following the instructions in the "How to obtain a Client ID" below the "Client ID" field.
    * Similarily, you can add Github by clicking "Social", click "Create Connection", find Github, and click "Continue", you will need to obtain the Client ID by following the instructions in the "How to obtain a Client ID" below the "Client ID" field. In the permissions section, it is recommended to check "Email address" so that Hypha can get the email address of the user.
    * Feel free to also customize the login page, and other settings in Auth0 dashboard.
 - Hypha also dependent on custom `roles` and `email` added in the JWT token by Auth0. You can add custom claims by installing a custom action in the login flow. 
    * Go to "Actions" in the Auth0 dashboard, and click "Create Action".
    * Choose "Create a new action", and choose "Login" as the trigger, then click "Create".
    * Give it a name, e.g. "Add Roles" and then in the code editor, replace the code with the following code:
    ```javascript
    exports.onExecutePostLogin = async (event, api) => {
      const namespace = 'https://amun.ai'; // replace with your own namespace, i.e. same as the AUTH0_NAMESPACE you set in the environment variables
      if (event.authorization) {
         api.idToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
         api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization.roles);
         api.accessToken.setCustomClaim(`${namespace}/email`, event.user.email);
         if(!event.user.email_verified){
         api.access.deny(`Access to ${event.client.name} is not allowed, please verify your email.`);
         }
      }
   };
    ```
    * Click "Deploy", and you can then drag and drop the action to the middle of the "Login" flow in the "Actions" tab. Make sure you have "Start" -> "Add Roles" -> "Complete" in the flow.
    * Now you should be able to see the `roles` and `email` in the JWT token when you login to Hypha.
