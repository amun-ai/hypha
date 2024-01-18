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
 AUTH0_CLIENT_ID=hMIMGeUvEHkVmi4KlGDSKfRPuGW43ypc # replace with your own value from the "Settings" tab
 AUTH0_DOMAIN=hypha.eu.auth0.com # replace with your own value from the "Settings" tab
 AUTH0_AUDIENCE=https://hypha.eu.auth0.com/api/v2/ # replace 'hypha.eu.auth0.com' to your own auth0 domain
 AUTH0_ISSUER=https://hypha.amun.ai/ # keep it or replace 'hypha.amun.ai' to any website you want to use as the issuer
 AUTH0_NAMESPACE=https://hypha.amun.ai/ # keep it or replace 'hypha.amun.ai' to any identifier you want to use as the namespace
 ```
 You can either set the environment variables in your system, or create a `.env` file in the root directory of Hypha, and add the above lines to the file.
 - Importantly, you also need to configure your own hypha server domain so Auth0 will allow it to login from your own domain. 
 For example, if you want to serve hypha server at https://my-company.com, you need to set the following in "Settings" tab:
    * scroll down to the "Allowed Callback URLs" section, and add the following URLs: https://my-company.com
    * scroll down to the "Allowed Logout URLs" section, and add the following URLs: https://my-company.com/public/apps/hypha-login/
    * scroll down to the "Allowed Web Origins" section, and add the following URLs: https://my-company.com
    * scroll down to the "Allowed Origins (CORS)" section, and add the following URLs: https://my-company.com
 For local development, you can also add `http://127.0.0.1:9000` to the above URLs, separated by comma. For example, "Allowed Callback URLs" can be `https://my-company.com,http://http://127.0.0.1:9000`.
 - Now you can start the hypha server (with the AUTH0 environment variables, via `python3 -m hypha.server --host=0.0.0.0 --port=9000`), and you should be able to test it by going to https://my-company.com/public/apps/hypha-login/ (replace with your own domain) or http://127.0.0.1:9000/public/apps/hypha-login.
 - By default, auth0 will provide a basic username-password-authentication which will store user information at auth0. You can also add other authentication providers (e.g. Google, Github) in the "Authenticaiton" tab of your application in Auth0 dashboard.
    * In order to add Google, click "Social", click "Create Connection", find Google/Gmail, and click "Continue", you will need to obtain the Client ID by following the instructions in the "How to obtain a Client ID" below the "Client ID" field.
    * Similarily, you can add Github by clicking "Social", click "Create Connection", find Github, and click "Continue", you will need to obtain the Client ID by following the instructions in the "How to obtain a Client ID" below the "Client ID" field. In the permissions section, it is recommended to check "Email address" so that Hypha can get the email address of the user.

Feel free to also customize the login page, and other settings in Auth0 dashboard.
