@echo off
echo Switching to Ruby 3.2...
set PATH=C:\Ruby32-x64\bin;%PATH%
echo Current Ruby version:
ruby --version
echo.
echo You can now run Jekyll commands.
echo Example: bundle install
echo Example: bundle exec jekyll serve
cmd /k